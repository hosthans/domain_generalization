from .resnet_ms import ResNet, BasicBlock, Bottleneck, model_urls, init_pretrained_weights

"""ResNet + FC"""


def resnet18_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


"""ResNet + FC + MixStyle"""


def resnet18_fc512_ms12_a0d1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=[512],
        dropout_p=None,
        mixstyle_layers=['layer1', 'layer2'],
        mixstyle_alpha=0.1,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet18_fc512_ms12_a0d2(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=[512],
        dropout_p=None,
        mixstyle_layers=['layer1', 'layer2'],
        mixstyle_alpha=0.2,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet18_fc512_ms12_a0d3(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=[512],
        dropout_p=None,
        mixstyle_layers=['layer1', 'layer2'],
        mixstyle_alpha=0.3,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet18_fc512_ms1_a0d1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=[512],
        dropout_p=None,
        mixstyle_layers=['layer1'],
        mixstyle_alpha=0.1,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet18_fc512_ms1_a0d2(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=[512],
        dropout_p=None,
        mixstyle_layers=['layer1'],
        mixstyle_alpha=0.2,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


"""ResNet + Freezing"""


def resnet18_fr123(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        frozen_layers=['layer1', 'layer2', 'layer3'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


"""ResNet + FC + Freezing"""


def resnet50_fc512_fr1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        frozen_layers=['layer1'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_fc512_fr12(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        frozen_layers=['layer1', 'layer2'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_fc512_fr123(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        frozen_layers=['layer1', 'layer2', 'layer3'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


"""ResNet Freezing + MixStyle"""


def resnet50_ms13_a0d1_fr13(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        mixstyle_layers=['layer1', 'layer3'],
        mixstyle_alpha=0.1,
        frozen_layers=['layer1', 'layer3'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_ms1_a0d1_fr13(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        mixstyle_layers=['layer1'],
        mixstyle_alpha=0.1,
        frozen_layers=['layer1', 'layer3'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_ms2_a0d1_fr12(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        mixstyle_layers=['layer2'],
        mixstyle_alpha=0.1,
        frozen_layers=['layer1', 'layer2'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_ms1_a0d1_fr1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        mixstyle_layers=['layer1'],
        mixstyle_alpha=0.1,
        frozen_layers=['layer1'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_ms12_a0d1_fr1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        mixstyle_layers=['layer1', 'layer2'],
        mixstyle_alpha=0.1,
        frozen_layers=['layer1'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


"""ResNet + FC + Freezing + MixStyle"""


def resnet18_fc512_ms12_a0d1_fr1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        mixstyle_layers=['layer1', 'layer2'],
        mixstyle_alpha=0.1,
        frozen_layers=['layer1'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet50_fc512_ms12_a0d1_fr1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        mixstyle_layers=['layer1', 'layer2'],
        mixstyle_alpha=0.1,
        frozen_layers=['layer1'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_fc512_ms2_a0d1_fr12(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        mixstyle_layers=['layer2'],
        mixstyle_alpha=0.1,
        frozen_layers=['layer1', 'layer2'],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
