'''
Test the DEM staging in scripts/stage_cslc_inputs.py (a script, loaded by
path): the OPERA DEM window, its nodata fill, and the --dem-source switch
'''
import importlib.util
import pathlib

import numpy as np
import pytest
from osgeo import gdal, osr

_PATH = pathlib.Path(__file__).parents[1] / 'scripts' / 'stage_cslc_inputs.py'
_spec = importlib.util.spec_from_file_location('stage_cslc_inputs', _PATH)
sci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sci)

GRANULE = 'S1A_IW_SLC__1SDV_20221016T015043_20221016T015111_045461_056FC0_6681'


def write_global_dem(path, nodata=-9999.0):
    '''A 0.1-degree EPSG:4326 DEM over [-119, 33, -117, 35] with a nodata and a
    NaN cell inside the window the tests ask for'''
    data = np.full((20, 20), 100.0, dtype=np.float32)
    data[5, 5] = nodata        # lon -118.45, lat 34.45
    data[6, 6] = np.nan        # lon -118.35, lat 34.35
    ds = gdal.GetDriverByName('GTiff').Create(str(path), 20, 20, 1,
                                              gdal.GDT_Float32)
    ds.SetGeoTransform([-119.0, 0.1, 0, 35.0, 0, -0.1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(data)
    ds = None
    return str(path)


def test_opera_window_fills_nodata_with_zero(tmp_path, monkeypatch):
    '''isce3 geo2rdr fails over nodata heights, so the window must carry 0
    where the DEM has no tile, and no nodata tag'''
    monkeypatch.setattr(sci, 'OPERA_DEM_VRT',
                        write_global_dem(tmp_path / 'global.tif'))
    dest = tmp_path / 'out' / 'dem.tif'
    sci._stage_dem_opera([-118.6, 34.2, -118.2, 34.6], dest)

    ds = gdal.Open(str(dest))
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    assert arr.shape == (4, 4)
    assert ds.GetGeoTransform()[0] == pytest.approx(-118.6)
    assert ds.GetGeoTransform()[3] == pytest.approx(34.6)
    assert band.GetNoDataValue() is None
    assert np.isfinite(arr).all()
    assert (arr == 0.0).sum() == 2
    assert (arr == 100.0).sum() == 14


def test_opera_source_reads_s3_through_vsis3(tmp_path, monkeypatch):
    '''The s3:// URL is opened as /vsis3/, so GDAL streams it from S3'''
    seen = {}

    def fake_translate(dest, src, **kwargs):
        seen['src'] = src
        raise RuntimeError('stop here')

    # _stage_dem_opera imports osgeo.gdal inside, so patch the module itself
    monkeypatch.setattr(gdal, 'Translate', fake_translate)
    with pytest.raises(RuntimeError, match='stop here'):
        sci._stage_dem_opera([-118.6, 34.2, -118.2, 34.6], tmp_path / 'd.tif')
    assert seen['src'] == '/vsis3/opera-dem/v1.1/EPSG4326/EPSG4326.vrt'


@pytest.mark.parametrize('source,called', [('opera', 'opera'),
                                           ('glo30', 'glo30')])
def test_stage_dem_routes_by_source(tmp_path, monkeypatch, source, called):
    calls = []

    def fake(kind):
        def stage(aoi, dest):
            calls.append((kind, aoi))
            write_global_dem(dest)
        return stage

    monkeypatch.setattr(sci, '_stage_dem_opera', fake('opera'))
    monkeypatch.setattr(sci, '_stage_dem_glo30', fake('glo30'))
    bbox = (-118.6, 34.2, -118.2, 34.6)
    sci.stage_dem(GRANULE, tmp_path, bbox=bbox, source=source)
    assert calls == [(called, list(bbox))]


def test_stage_dem_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="Unknown DEM source 'srtm'"):
        sci.stage_dem(GRANULE, tmp_path, bbox=(-118.6, 34.2, -118.2, 34.6),
                      source='srtm')


@pytest.mark.parametrize('argv,expected', [
    ([], 'opera'),
    (['--dem-source', 'glo30'], 'glo30'),
])
def test_cli_dem_source(tmp_path, monkeypatch, argv, expected):
    seen = {}
    monkeypatch.setattr(sci, 'stage_dem',
                        lambda *a, **kw: seen.update(kw))
    sci.main(['dem', GRANULE, '-i', str(tmp_path),
              '--bbox', '-118.6', '34.2', '-118.2', '34.6', *argv])
    assert seen['source'] == expected


def test_cli_rejects_unknown_dem_source(tmp_path):
    with pytest.raises(SystemExit) as err:
        sci.main(['dem', GRANULE, '-i', str(tmp_path), '--dem-source', 'srtm'])
    assert err.value.code == 2
