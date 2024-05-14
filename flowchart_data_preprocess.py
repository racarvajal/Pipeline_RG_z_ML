#!/usr/bin/env python

import schemdraw
from schemdraw import flow
import schemdraw.elements as elm
import matplotlib as mpl
import matplotlib.pyplot as plt
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

with schemdraw.Drawing(show=False) as d:
    # d.config(unit=.75)
    init       = flow.Start(h=1.5, w=2.75).at((0, 1.5)).label('CatWISE2020\nSOURCES')# .drop('S')
    cat_AW     = flow.Data(h=1, w=2).at((3.50, -2.0)).label('AW')
    cat_2MASS  = flow.Data(h=1, w=2).at((5.75, -3.5)).label('2MASS')
    cat_PS1    = flow.Data(h=1, w=2).at((8.00, -5.0)).label('PS1')
    cat_radio  = flow.Data(h=1, w=2).at((-4, -2.5)).label('LoTSS\nVLAS82')
    cat_class  = flow.Data(h=1, w=2).at((-4, -4.5)).label('MQC\nSDSS')

    x_match = elm.EncircleBox([cat_2MASS, cat_PS1, cat_AW, cat_radio, cat_class], padx=1, pady=.3).linestyle('-').linewidth(2).label('CROSS-MATCH', halign='left', ofst=(5.0, 0.0))
    compiled    = flow.Box(h=1.0, w=2.5).at((0, -7.0)).label('COMPILE\nFEATURES')
    elm.Wire('-|', arrow='->').at(cat_AW.W).to(compiled.N).label('$\mathtt{W3mag}$\n$\mathtt{W4mag}$', loc='top', ofst=(0.5, -.1))# .label(r'$\mathtt{band\_num}$', loc='left')
    elm.Wire('-|', arrow='->').at(cat_2MASS.W).to(compiled.N).label('$\mathtt{Jmag, Hmag,}$\n$\mathtt{Kmag}$', loc='top', ofst=(1, -.1))
    elm.Wire('-|', arrow='->').at(cat_PS1.W).to(compiled.N).label('$\mathtt{gmag, imag,}$\n$\mathtt{zmag, ymag}$', loc='top', ofst=(2, -.1))
    elm.Wire('-|', arrow='->').at(cat_radio.E).to(compiled.N).label(r'$\mathtt{radio\_detect}$', loc='top')
    elm.Wire('-|', arrow='->').at(cat_class.E).to(compiled.N).label(r'$\mathtt{class,Z}$', loc='top')
    elm.Wire('-', arrow='->').at(init.S).to(compiled.N).label('$\mathtt{W1mproPM}$\n$\mathtt{W2mproPM}$', loc='left', ofst=(-0.2, 3.0))
    impute     = flow.Box(h=1.0, w=2.5).at((0, -9.0)).label('IMPUTE\nMAGNITUDES')
    elm.Wire('-', arrow='->').at(compiled.S).to(impute.N).label('$\mathtt{band\_num}$', loc='right', ofst=(0.1, 0))
    colours    = flow.Box(h=1.0, w=2.5).at((0, -11.0)).label('COMPUTE\nCOLOURS')
    elm.Wire('-', arrow='->').at(impute.S).to(colours.N)
    std        = flow.Box(h=1.0, w=3.5).at((0, -13.0)).label('STANDARDISATION')
    norm       = flow.Box(h=1.0, w=2.5).at((0, -15.0)).label('POWER\nTRANSFORM')
    elm.Wire('-', arrow='->').at(std.S).to(norm.N)
    data_trf = elm.EncircleBox([std, norm], padx=-.3, pady=.1).linestyle('-').linewidth(2).label('DATA\nTRANSFORMATION', ofst=(3.0, -3.3), rotate=270)
    data_trf.at((0, -14))
    elm.Wire('-', arrow='->').at(colours.S).to(std.N)
    correlat   = flow.Box(h=1.0, w=2.8).at((0, -17.0)).label('CORRELATION\nFACTOR')
    pps        = flow.Box(h=1.0, w=2.8).at((0, -19.0)).label('PREDICTIVE\nPOWER SCORE')
    elm.Wire('-', arrow='->').at(correlat.S).to(pps.N)
    feat_sel = elm.EncircleBox([correlat, pps], padx=.1, pady=-.1).linestyle('-').linewidth(2).label('FEATURE\nSELECTION', ofst=(2.9, -3.4), rotate=270)
    feat_sel.at((0, -18.0))
    elm.Wire('-', arrow='->').at(norm.S).to(correlat.N)
    final_cat = flow.Start(h=1.5, w=2.5).at((0, -21.0)).label('FINAL\nCATALOGUE')
    elm.Wire('-', arrow='->').at(pps.S).to(final_cat.N)
    models    = flow.Start(h=1.5, w=2.5).at((0, -23.0)).label('MODEL\nTRAINING')
    elm.Wire('-', arrow='->').at(final_cat.S).to(models.N)

    final_ghost = flow.Start(h=1.0, w=2.5).at((0, -28.5))
    # d.draw(show=True)
    # d.save(gv.plots_path + 'flowchart_data_preprocess.pdf')
print('EOF')