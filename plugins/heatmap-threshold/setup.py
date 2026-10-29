from setuptools import find_packages, setup


setup(
    name='girder-threshold-mask',
    version='0.2.0',
    description='Browser-side cortex and medulla contour annotations for HistomicsUI',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['girder>=3,<5', 'histomicsui'],
    entry_points={'girder.plugin': [
        'threshold_mask = girder_threshold_mask:ThresholdMaskPlugin',
    ]},
    zip_safe=False,
    data_files=[
        (
            "share/girder/static/built/plugins/threshold_mask",
            [
                "girder_threshold_mask/web_client/contour-worker.js",
                "girder_threshold_mask/web_client/polygon-clipping.umd.min.js",
                "girder_threshold_mask/web_client/polygon-clipping.LICENSE",
            ],
        ),
    ],
)
