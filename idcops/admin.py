# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.contrib import admin
from django.db import models
from django.apps import apps
from idcops.models import User
from django.utils.translation import ugettext, ugettext_lazy as _
from django.contrib.auth.models import Group
from django.contrib.auth.admin import UserAdmin, GroupAdmin
from idcops.lib.utils import nature_field_name
# Register your models here.

admin.AdminSite.site_header = '漓泉运维管理后台'
admin.AdminSite.site_title = '数据中心运维平台'

try:
    app_models = apps.get_models()
except BaseException:
    app_models = None

if app_models:
    exclude_fields = ['creator', 'actived', 'deleted', 'modified', 'operator']
    for model in app_models:
        if not admin.site.is_registered(model):
            opts = model._meta
            list_filter = []
            search_fields = []
            for f in opts.fields:
                if isinstance(
                        f, (models.BooleanField, models.NullBooleanField)):
                    list_filter.append(f.name)
                if getattr(f, 'flatchoices', None):
                    list_filter.append(f.name)
                if isinstance(
                    f,
                    (models.CharField,
                     models.SlugField,
                     models.TextField)):
                    search_fields.append(f.name)
            exclude_fields.extend(list_filter)
            options = {
                'list_display': [
                    f.name for f in opts.fields if f.name not in exclude_fields],
                'list_filter': list_filter,
                'list_display_links': [
                    nature_field_name(model)],
                'search_fields': search_fields,
                'list_per_page': 20,
            }
            try:
                admin.site.register(model, **options)
            except BaseException:
                pass

if admin.site.is_registered(User):
    admin.site.unregister(User)

    @admin.register(User)
    class UserAdmin(UserAdmin):
        fieldsets = (
            (None, {'fields': ('username', 'password')}),
            (_('Personal info'), {'fields': ('first_name', 'email', 'upper',
                                             'onidc', 'mobile', 'mark', 'last_name', 'actived', 'avatar', 'slaveidc')}),
            (_('Permissions'), {'fields': ('is_active', 'is_staff', 'is_superuser',
                                           'groups', 'user_permissions')}),
            (_('Important dates'), {'fields': ('last_login', 'date_joined')}),
        )
        add_fieldsets = (
            (None, {
                'classes': (
                    'wide',), 'fields': (
                    'username', 'password1', 'password2', 'first_name', 'email', 'onidc', 'slaveidc'), }), )

        filter_horizontal = ('groups', 'user_permissions', 'slaveidc')

if not admin.site.is_registered(Group):
    # admin.site.unregister(Group)
    @admin.register(Group)
    class GroupAdmin(GroupAdmin):
        pass
