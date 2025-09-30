# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys

from django import forms
from django.utils.six import text_type
from django.utils.html import format_html
from django.utils.text import capfirst, get_text_list
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.forms import UserCreationForm
from idcops.models import (
    Option, Rack, Comment, User, Client, Device,
    Idc,
    Document, Configure, Rextend, Unit, Pdu
)

from idcops.lib.utils import can_create, shared_queryset

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except Exception as e:
    pass

STATICROOT = '/static/idcops/'


class CalendarMedia(object):

    class Media:
        extend = True
        css = {
            'all': (
                '/static/idcops/css/daterangepicker.min.css',
            )
        }

        js = (
            '/static/idcops/js/moment.min.js',
            '/static/idcops/js/daterangepicker.min.js',
        )


class Select2Media(object):

    class Media:
        css = {
            'all': (
                '/static/idcops/css/select2.min.css',
            )
        }

        js = (
            '/static/idcops/js/select2.min.js',
            '/static/idcops/js/i18n/zh-CN.js',
        )


class CheckUniqueTogether(forms.ModelForm):

    def get_unique_together(self):
        unique_together = self.instance._meta.unique_together
        for field_set in unique_together:
            return field_set
        return None

    def clean(self):
        # self.validate_unique()
        cleaned_data = super(CheckUniqueTogether, self).clean()
        unique_fields = self.get_unique_together()
        if isinstance(unique_fields, (list, tuple)):
            unique_filter = {}
            instance = self.instance
            model_name = instance._meta.verbose_name
            for unique_field in unique_fields:
                field = instance._meta.get_field(unique_field)
                if field.editable and unique_field in self.fields:
                    unique_filter[unique_field] = cleaned_data.get(
                        unique_field)
                else:
                    unique_filter[unique_field] = getattr(
                        instance, unique_field)
            for k, v in unique_filter.items():
                if not v:
                    return
            existing_instances = type(instance).objects.filter(
                **unique_filter).exclude(pk=instance.pk)
            if existing_instances:
                field_labels = [
                    instance._meta.get_field(f).verbose_name
                    for f in unique_fields
                ]
                field_labels = text_type(get_text_list(field_labels, _('and')))
                msg = _("%(model_name)s with this %(field_labels)s already exists.") % {
                    'model_name': model_name, 'field_labels': field_labels, }
                for unique_field in unique_fields:
                    if unique_field in self.fields:
                        self.add_error(unique_field, msg)


class FormBaseMixin(Select2Media, CheckUniqueTogether):
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(FormBaseMixin, self).__init__(*args, **kwargs)
        if 'mark' in self.fields:
            self.fields['mark'].widget = forms.HiddenInput()
        if self.user is not None:
            onidc_id = self.user.onidc_id
            effective = {
                'onidc_id': onidc_id,
                'deleted': False,
                'actived': True}
            for field_name in self.fields:
                field = self.fields.get(field_name)
                if isinstance(
                    field,
                    (forms.fields.SlugField,
                     forms.fields.CharField)):
                    self.fields[field_name].widget.attrs.update(
                        {'autocomplete': "off"})
                if isinstance(field, forms.fields.DateTimeField):
                    self.fields[field_name].widget.attrs.update(
                        {'data-datetime': "true"})
                if isinstance(field.widget, forms.widgets.Textarea):
                    self.fields[field_name].widget.attrs.update({'rows': "3"})
                if isinstance(field, (
                        forms.models.ModelChoiceField,
                        forms.models.ModelMultipleChoiceField)):
                    fl = ''
                    if getattr(field.queryset.model, 'mark', False):
                        field.queryset = shared_queryset(
                            field.queryset, onidc_id)
                        if field.queryset.model is Option:
                            _prefix = self._meta.model._meta.model_name
                            _postfix = field_name.capitalize()
                            flag = _prefix.capitalize() + '-' + _postfix
                            fl = flag
                            field_initial = field.queryset.filter(
                                master=True, flag=flag)
                            if field_initial.exists():
                                field.initial = field_initial.first()
                    else:
                        field.queryset = field.queryset.filter(**effective)
                    mn = field.queryset.model._meta
                    if can_create(mn, self.user) and fl:
                        fk_url = format_html(
                            ''' <a title="点击添加一个 {}"'''
                            ''' href="/new/{}/?flag={}">'''
                            '''<i class="fa fa-plus"></i></a>'''.format(
                                field.label, mn.model_name, fl))
                    elif can_create(mn, self.user) and not fl:
                        fk_url = format_html(
                            ''' <a title="点击添加一个 {}"'''
                            ''' href="/new/{}">'''
                            '''<i class="fa fa-plus"></i></a>'''.format(
                                field.label, mn.model_name))
                    else:
                        fk_url = ''
                    field.help_text = field.help_text + fk_url
                self.fields[field_name].widget.attrs.update(
                    {'class': "form-control"})


class UserNewForm(Select2Media, UserCreationForm):
    class Meta(FormBaseMixin, UserCreationForm.Meta):
        model = User
        fields = (
            "username",
            "first_name",
            "email",
            "mobile",
            "groups",
            "slaveidc")

    def __init__(self, *args, **kwargs):
        if 'user' in kwargs:
            self.user = kwargs.pop('user', None)
        super(UserNewForm, self).__init__(*args, **kwargs)
        if self._meta.model.USERNAME_FIELD in self.fields:
            self.fields[self._meta.model.USERNAME_FIELD].widget.attrs.update(
                {'autofocus': True})
        for field in self.fields:
            self.fields[field].widget.attrs.update(
                {'autocomplete': "off", 'class': "form-control"})
        if not self.user.is_superuser:
            self.fields['slaveidc'].queryset = self.user.slaveidc.all()
            self.fields['groups'].queryset = self.user.groups.all()


class UserEditForm(Select2Media, forms.ModelForm):
    class Meta:
        model = User
        fields = (
            "username",
            "first_name",
            "email",
            "mobile",
            "upper",
            "groups",
            "onidc",
            "slaveidc",
            'user_permissions')

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(UserEditForm, self).__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                'autocomplete': "off", 'class': "form-control"
            })
        if self.user and not self.user.is_superuser:
            self.fields['onidc'].queryset = self.user.slaveidc.all()
            self.fields['slaveidc'].queryset = self.user.slaveidc.all()
            self.fields['groups'].queryset = self.user.groups.all()
            self.fields['upper'].queryset = self._meta.model.objects.filter(
                upper=self.user)
            self.fields['user_permissions'].queryset = self.user.user_permissions.all()


class OptionForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Option
        fields = [
            'flag',
            'text',
            'description',
            'master',
            'color',
            'parent',
            'mark']

    def __init__(self, *args, **kwargs):
        self.flag = kwargs.pop('flag', None)
        super(OptionForm, self).__init__(*args, **kwargs)
        self.fields['flag'].choices = self._meta.model().choices_to_field
        if self.flag:
            self.fields['flag'].initial = self.flag


class IdcForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Idc
        fields = [
            'name', 'desc', 'emailgroup', 'managers', 'address', 'duty', 'tel'
        ]

    def __init__(self, *args, **kwargs):
        super(IdcForm, self).__init__(*args, **kwargs)
        if self.user and self.user.is_superuser:
            self.fields['managers'].queryset = User.objects.filter(
                actived=True, is_active=True
            )


class ClientForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Client
        fields = ['name', 'style', 'sales', 'kf', 'tags']


class RackNewForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Rack
        fields = ['name', 'cname', 'unitc', 'pduc', 'tags', 'actived']


    def __init__(self, *args, **kwargs):
        super(RackNewForm, self).__init__(*args, **kwargs)
        self.fields['actived'].initial = False
        self.fields['actived'].widget = forms.HiddenInput()


class RextendNewForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Rextend
        fields = ['rack', 'user', 'ups1', 'ups2', 'temperature', 'humidity']

    def __init__(self, rack_id=None, *args, **kwargs):
        super(RextendNewForm, self).__init__(*args, **kwargs)
        self.fields['rack'].widget = forms.HiddenInput()
        self.fields['user'].widget = forms.HiddenInput()
        if rack_id is not None:
            self.fields['rack'].initial = self.fields['rack'].queryset.get(
                pk=rack_id)
            self.fields['user'].initial = self.fields['rack'].queryset.get(
                pk=rack_id).User


class UnitForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Unit
        fields = ['rack', 'name']


class PduForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Pdu
        fields = ['rack', 'name']


class RackEditForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Rack
        fields = ['name', 'cname', 'tags']


class OnlineNewForm(CalendarMedia, FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Device
        fields = [
            'rack', 'user', 'created', 'style', 'name',
             'ipaddr', 'model', 'sn', 'units', 'pdus', 'tags'
        ]

    def __init__(self, *args, **kwargs):
        rack_id = kwargs.pop("rack_id", None)
        super(OnlineNewForm, self).__init__(*args, **kwargs)
        self.fields['rack'].empty_label = u'请选择一个数据'
        self.fields['rack'].queryset = self.fields['rack'].queryset.order_by(
            'name')
        if rack_id is None:
            self.fields['User'].queryset = self.fields['User'].queryset.none()
            self.fields['style'].queryset = self.fields['style'].queryset.none()
            self.fields['units'].queryset = self.fields['units'].queryset.none()
            self.fields['pdus'].queryset = self.fields['pdus'].queryset.none()
            self.fields['tags'].queryset = self.fields['tags'].queryset.none()
        else:
            self.fields['style'].empty_label = None
            rack = self.fields['rack'].queryset.get(pk=rack_id)
            self.fields['rack'].initial = rack_id
            try:
                onidc_id = self.user.onidc_id
                name = Device.objects.filter(
                    onidc_id=onidc_id).order_by('-pk').first().name
            except:
                name = 'IS020123456-0000'
            try:
                pre, lnk, ext = name.rpartition('-')
                ext = "%05d" %(int(ext)+1)
                self.fields['name'].initial = pre+lnk+ext
            except:
                self.fields['name'].initial = name
            try:
                self.fields['User'].initial = rack.client_id
            except BaseException:
                pass
            self.fields['units'].queryset = self.fields['units'].queryset.filter(
                rack_id=rack_id).order_by('name')
            self.fields['pdus'].queryset = self.fields['pdus'].queryset.filter(
                rack_id=rack_id).order_by('pk')


class OnlineEditForm(CalendarMedia, FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Device
        fields = [
            'rack', 'user', 'style', 'ipaddr',
            'name', 'sn', 'model',
            'units', 'pdus', 'tags'
        ]

    def __init__(self, *args, **kwargs):
        rack_id = kwargs.pop("rack_id", None)
        super(OnlineEditForm, self).__init__(*args, **kwargs)
        if not rack_id:
            rack_id = self.instance.rack_id
        self.fields['rack'].queryset = self.fields['rack'].queryset.order_by(
            'name')
        #rack = self.fields['rack'].queryset.get(pk=rack_id)
        self.fields['units'].queryset = self.fields['units'].queryset.filter(
            rack_id=rack_id) | self.instance.units.all()
        self.fields['pdus'].queryset = self.fields['pdus'].queryset.filter(
            rack_id=rack_id).order_by("pk") | self.instance.pdus.all()


class CommentNewForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content']


class DocumentForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Document
        fields = ['title', 'body', 'category', 'status', 'tags']


    class Media:
        extend = True
        css = {
            'all': (
                '/static/idcops/dist/summernote.css',
            )
        }

        js = (
            '/static/idcops/dist/summernote.js',
            '/static/idcops/dist/lang/summernote-zh-CN.min.js',
        )


class ConfigureNewForm(FormBaseMixin, forms.ModelForm):
    class Meta:
        model = Configure
        fields = ['mark', 'content']
