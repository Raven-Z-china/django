# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.dispatch import receiver
from django.db.models import signals
from idcops.lib.utils import get_content_type_for_model, fields_for_model
from idcops.models import (
    Device, Online, Offline,
    Rack, Unit, Pdu, User, Configure
)


def pdus_units_changed(sender, **kwargs):
    model = kwargs.pop('model', None)
    pk_set = kwargs.pop('pk_set', None)
    action = kwargs.pop('action', None)
    if action == 'pre_add':
        objects = model.objects.filter(pk__in=pk_set)
        for obj in objects:
            if not obj.actived:
                raise('actived current is `false`')
    if action == 'post_add':
        model.objects.filter(pk__in=pk_set).update(actived=False)
    if action in ["pre_remove", "post_remove"]:
        model.objects.filter(pk__in=pk_set).update(actived=True)


signals.m2m_changed.connect(
    pdus_units_changed, sender=Device.units.through,
    dispatch_uid='when_device_units_changed'
)

signals.m2m_changed.connect(
    pdus_units_changed, sender=Device.pdus.through,
    dispatch_uid='when_device_pdus_changed'
)


@receiver(signals.post_delete, sender=Device)
def update_units_pdus(**kwargs):
    instance = kwargs.pop('instance', None)
    instance.units.all().update(actived=True)
    instance.pdus.all().update(actived=True)


@receiver(signals.post_save, sender=Rack, dispatch_uid='rack_created_tasks')
def rack_created_tasks(instance, created, **kwargs):
    onidc_id = instance.onidc_id
    creator_id = instance.creator_id
    if created and onidc_id:
        units = []
        pdus = []
        for unit in range(1, int(instance.unitc + 1)):
            name = str(unit).zfill(2)
            units.append(Unit(onidc_id=onidc_id, name=name,
                              rack=instance, creator_id=creator_id))
        Unit.objects.bulk_create(units)
        for pdu in range(1, int((instance.pduc + 2) / 2)):
            pdus.append(Pdu(onidc_id=onidc_id, creator_id=creator_id,
                            rack=instance, name='A' + str(pdu)))
            pdus.append(Pdu(onidc_id=onidc_id, creator_id=creator_id,
                            rack=instance, name='B' + str(pdu)))
        Pdu.objects.bulk_create(pdus)
    if not created and instance.client is not None:
        Unit.objects.filter(rack=instance).update(client=instance.client)
        Pdu.objects.filter(rack=instance).update(client=instance.client)


@receiver(signals.post_save, sender=User, dispatch_uid='initial_user_config')
def initial_user_config(instance, created, **kwargs):
    if created:
        import json
        from django.apps import apps
        models = apps.get_app_config('idcops').get_models()
        exclude = ['onidc', 'deleted', 'mark']
        configures = []
        for model in models:
            fds = [f for f in fields_for_model(model) if f not in exclude]
            _fields = getattr(model._meta, 'list_display', fds)
            fields = _fields if isinstance(_fields, list) else fds
            content = {'list_only_date': 1, 'list_display': fields}
            config = {
                'onidc': instance.onidc,
                'creator': instance,
                'mark': 'list',
                'content_type': get_content_type_for_model(model),
                'content': json.dumps(content),
            }
            configures.append(Configure(**config))
        Configure.objects.bulk_create(configures)
