#!/usr/bin/env python

import schemdraw
from schemdraw import flow
import schemdraw.elements as elm
import matplotlib as mpl
import matplotlib.pyplot as plt
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

save_plot_flag = False

with schemdraw.Drawing(show=False) as d:
    init           = flow.Start(h=1.75, w=2.75).at((0, 0)).label('SOURCE\nFROM\nCATALOGUE')
    AGN_gal_model  = flow.Decision(h=2.25, w=4.00, S='AGN', E='Non AGN').at((0, -2.5)).label('AGN\nSELECTION')
    elm.Wire('-', arrow='->').at(init.S).to(AGN_gal_model.N)
    rAGN_model     = flow.Decision(h=2.25, w=4.00, S='Radio AGN', E='Non-radio\nAGN').at((0, -5.5)).label('AGN RADIO\nDETECTION\nSELECTION')
    elm.Wire('-', arrow='->').at(AGN_gal_model.S).to(rAGN_model.N)
    discarded      = flow.StateEnd(r=1.25).at((5.5, -5.40)).label('DISCARDED\nSOURCE')
    elm.Wire('-', arrow='->').at(rAGN_model.E).to(discarded.W)
    elm.Wire('-|', arrow='->').at(AGN_gal_model.E).to(discarded.N)
    z_rAGN_model   = flow.Box(h=1.5, w=2.5).at((0, -8.5)).label('REDSHIFT\nESTIMATION')
    elm.Wire('-', arrow='->').at(rAGN_model.S).to(z_rAGN_model.N)
    final_state    = flow.StateEnd(r=1.30).at((0, -10.75)).label('PREDICTED\nRADIO AGN\nw/REDSHIFT')
    elm.Wire('-', arrow='->').at(z_rAGN_model.S).to(final_state.N)

    final_ghost    = flow.Start(h=1.0, w=2.5).at((0, -18.0))
    # d.draw(show=True)
    if save_plot_flag:
        d.save(gv.plots_path + 'flowchart_pipeline_initial_steps.pdf')

with schemdraw.Drawing(show=False) as e:
    init           = flow.Start(h=1.75, w=2.75).at((0, 0)).label('SOURCE\nFROM\nCATALOGUE')
    AGN_gal_model  = flow.Decision(h=2.25, w=4.00, S='Predicted\nas AGN', E='Predicted\nas galaxy').at((0, -2.5)).label('AGN\nCLASSIFICATION\nMODEL')
    elm.Wire('-', arrow='->').at(init.S).to(AGN_gal_model.N)
    rAGN_model     = flow.Decision(h=2.25, w=4.00, S='Predicted\nas radio', E='Predicted\nas no radio').at((0, -5.5)).label('RADIO\nDETECTION\nMODEL')
    elm.Wire('-', arrow='->').at(AGN_gal_model.S).to(rAGN_model.N)
    discarded      = flow.StateEnd(r=1.25).at((5.5, -5.40)).label('DISCARDED\nSOURCE')
    elm.Wire('-', arrow='->').at(rAGN_model.E).to(discarded.W)
    elm.Wire('-|', arrow='->').at(AGN_gal_model.E).to(discarded.N)
    z_rAGN_model   = flow.Box(h=1.5, w=2.5).at((0, -8.75)).label('REDSHIFT\nPREDICTION\nMODEL')
    elm.Wire('-', arrow='->').at(rAGN_model.S).to(z_rAGN_model.N)
    final_state    = flow.StateEnd(r=1.30).at((0, -10.75)).label('PREDICTED\nRADIO AGN\nw/REDSHIFT')
    elm.Wire('-', arrow='->').at(z_rAGN_model.S).to(final_state.N)

    final_ghost    = flow.Start(h=1.0, w=2.5).at((0, -18.0))
    # e.draw(show=True)
    if save_plot_flag:
        e.save(gv.plots_path + 'flowchart_pipeline_ML_steps.pdf')

print('EOF')
