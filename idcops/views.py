# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from django.shortcuts import render

# Create your views here.
from django.apps import apps
from django.views.generic import View, TemplateView
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.core.urlresolvers import reverse_lazy
from django.contrib import messages
from django.contrib.auth.views import (
    LoginView, LogoutView, PasswordResetView,
    PasswordResetDoneView, PasswordResetConfirmView,
    PasswordResetCompleteView, PasswordChangeDoneView,
    PasswordChangeView
)
from django.template.loader import render_to_string
from django.utils.translation import ugettext as _
from django.utils.encoding import force_text
from django.utils.functional import cached_property
from django.utils.module_loading import import_string

from idcops.lib.utils import shared_queryset
from idcops.mixins import BaseRequiredMixin
from idcops.list import ListModelView
from idcops.detail import DetailModelView
from idcops.edit import NewModelView, EditModelView
from idcops.models import (
    Option, Rack, Device, Online, Offline,
    Syslog, ContentType
)


login = LoginView.as_view(template_name='accounts/login.html')
logout = LoginView.as_view(template_name='accounts/logout.html')

password_reset = PasswordResetView.as_view(
    template_name='accounts/password_reset_form.html',
    email_template_name='accounts/password_reset_email.html',
    subject_template_name='accounts/password_reset_subject.txt',
)

password_reset_done = PasswordResetDoneView.as_view(
    template_name='accounts/password_reset_done.html'
)

reset = PasswordResetConfirmView.as_view(
    template_name='accounts/password_reset_confirm.html'
)

reset_done = PasswordResetCompleteView.as_view(
    template_name='accounts/password_reset_complete.html'
)


class PasswordChangeView(BaseRequiredMixin, PasswordChangeView):
    template_name = 'accounts/password_change_form.html'
    success_url = reverse_lazy('idcops:index')

password_change = PasswordChangeView.as_view()

password_change_done = PasswordChangeDoneView.as_view(
    template_name='accounts/password_change_done.html'
)


class SummernoteUploadAttachment(BaseRequiredMixin, View):
    def __init__(self):
        super(SummernoteUploadAttachment, self).__init__()

    def get(self, request, *args, **kwargs):
        return JsonResponse({
            'status': 'false',
            'message': _('Only POST method is allowed'),
        }, status=400)

    def post(self, request, *args, **kwargs):
        if not request.FILES.getlist('files'):
            return JsonResponse({
                'status': 'false',
                'message': _('No files were requested'),
            }, status=400)

        # remove unnecessary CSRF token, if found
        kwargs = request.POST.copy()
        kwargs.pop("csrfmiddlewaretoken", None)

        try:
            attachments = []

            for file in request.FILES.getlist('files'):

                # create instance of appropriate attachment class
                klass = get_attachment_model()
                attachment = klass()

                attachment.file = file
                attachment.name = file.name

                if file.size > config['attachment_filesize_limit']:
                    return JsonResponse({
                        'status': 'false',
                        'message': _('File size exceeds the limit allowed and cannot be saved'),
                    }, status=400)

                # calling save method with attachment parameters as kwargs
                attachment.save(**kwargs)
                attachments.append(attachment)

            return HttpResponse(render_to_string('document/upload_attachment.json', {
                'attachments': attachments,
            }), content_type='application/json')
        except IOError:
            return JsonResponse({
                'status': 'false',
                'message': _('Failed to save attachment'),
            }, status=500)


class IndexView(BaseRequiredMixin, TemplateView):

    template_name = 'index.html'

    def make_years(self, queryset):
        years = queryset.datetimes('created', 'month')[:12]
        return years

    def make_device_dynamic_change(self):
        content_type=ContentType.objects.get_for_model(Device)
        logs = Syslog.objects.filter(
            onidc_id=self.onidc_id, content_type=content_type)
        data = {}
        data['categories'] = [m.strftime("%Y-%m") for m in self.make_years(logs)]
        data['moveup'] = []
        data['moving'] = []
        data['movedown'] = []
        for y in self.make_years(logs):
            nlogs = logs.filter(created__year=y.year, created__month=y.month)
            moving = nlogs.filter(
                message__contains='["units"]', action_flag="修改").exclude(
                content__contains='{"units": [[],').count()
            data['moving'].append(moving)
            moveup = nlogs.filter(action_flag="新增").count()
            data['moveup'].append(moveup)
            cancel_movedown = nlogs.filter(action_flag="取消下架").count()
            movedown = nlogs.filter(action_flag="下架").count()
            data['movedown'].append(movedown-cancel_movedown)
        return data

    def make_rack_dynamic_change(self):
        content_type=ContentType.objects.get_for_model(Rack)
        logs = Syslog.objects.filter(
            onidc_id=self.onidc_id, content_type=content_type)
        data = {}
        data['categories'] = [m.strftime("%Y-%m") for m in self.make_years(logs)]
        data['renew'] = []
        data['release'] = []
        for y in self.make_years(logs):
            nlogs = logs.filter(created__year=y.year, created__month=y.month)
            data['renew'].append(nlogs.filter(action_flag="分配模型").count())
            data['release'].append(nlogs.filter(action_flag="释放机柜").count())
        return data

    def make_rack_statistics(self):
        data = []
        robjects = Rack.objects.filter(onidc_id=self.onidc_id, actived=True)
        keys = Option.objects.filter(
            flag__in=['Rack-Style','Rack-Status'],
            actived=True)
        keys = shared_queryset(keys, self.onidc_id)
        for k in keys:
            d = []
            query = {}
            query[k.flag.split('-')[1].lower()] = k
            c = robjects.filter(**query).count()
            if c > 0:
                d.append(force_text(k))
                d.append(c)
            if d:
                data.append(d)
        return data

    def make_online_statistics(self):
        data = []
        dobjects = Online.objects.filter(onidc_id=self.onidc_id)
        keys = Option.objects.filter(flag__in=['Device-Style', 'Device-Tags'])
        keys = shared_queryset(keys, self.onidc_id)
        for k in keys:
            d = []
            if k.flag == 'Device-Style':
                c = dobjects.filter(style=k).count()
            else:
                c = dobjects.filter(tags__in=[k]).count()
            if c > 0:
                d.append(force_text(k))
                d.append(c)
            if d:
                data.append(d)
        return data

    def make_state_items(self):
        state_items = [
            {
                'model_name': app._meta.model_name,
                'verbose_name': app._meta.verbose_name,
                'icon': app._meta.icon,
                'icon_color': 'bg-' + app._meta.icon_color,
                'level': app._meta.level,
                'metric': app._meta.metric,
                'count': app.objects.filter(
                    onidc=self.request.user.onidc).filter(
                    **app._meta.default_filters).count(),
            } for app in apps.get_app_config('idcops').get_models() if getattr(
                app._meta,
                'dashboard')]
        return state_items

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        context['state_items'] = self.make_state_items()
        context['online_statistics'] = self.make_online_statistics()
        context['device_dynamic_change'] = self.make_device_dynamic_change()
        context['rack_statistics'] = self.make_rack_statistics()
        context['rack_dynamic_change'] = self.make_rack_dynamic_change()
        return context


class ProfileView(BaseRequiredMixin, TemplateView):
    template_name = 'accounts/profile.html'

    def get(self, *args, **kwargs):
        #messages.success(self.request, u"accounts/profile.html")
        return super(ProfileView, self).get(*args, **kwargs)


