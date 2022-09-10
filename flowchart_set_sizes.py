#!/usr/bin/env python

# File to create flowchart
# for splitting of sets and
# sub-sets in both HETDEX
# and Stripe 82 fields

import schemdraw
from schemdraw import flow
import global_variables as gv

save_plot_flag = False

# HETDEX
with schemdraw.Drawing(fontsize=11) as d:
    d += (HETDEX := flow.Start(w=3, h=1.5).label('HETDEX Field\n6,729,647'))
    d += schemdraw.elements.lines.Gap().at(HETDEX.S)

    d += (Labelled := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Labelled\n83,409'))
    d += (Unlabelled := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Unlabelled\n6,646,238'))
    d += flow.Arrow().length(d.unit/3).at(HETDEX.S).to(Labelled.N)
    d += flow.Arrow().length(d.unit/3).at(HETDEX.S).to(Unlabelled.N)
    d += schemdraw.elements.lines.Gap().at(Labelled.S)

    d += (Tr_Te_Ca := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Train+Test+\nCalibration\n66,727'))
    d += (Validation := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Validation\n16,682'))
    d += flow.Arrow().length(d.unit/3).at(Labelled.S).to(Tr_Te_Ca.N)
    d += flow.Arrow().length(d.unit/3).at(Labelled.S).to(Validation.N)
    d += schemdraw.elements.lines.Gap().at(Tr_Te_Ca.S)

    d += (Train := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Train\n53,381'))
    d += (Te_Ca := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Test+Calibration\n13,346'))
    d += flow.Arrow().length(d.unit/3).at(Tr_Te_Ca.S).to(Train.N)
    d += flow.Arrow().length(d.unit/3).at(Tr_Te_Ca.S).to(Te_Ca.N)
    d += schemdraw.elements.lines.Gap().at(Te_Ca.S)

    d += (Test := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Test\n6,673'))
    d += (Calibration := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Calibration\n6,673'))
    d += flow.Arrow().length(d.unit/3).at(Te_Ca.S).to(Test.N)
    d += flow.Arrow().length(d.unit/3).at(Te_Ca.S).to(Calibration.N)
      
    # d.draw()
    if save_plot_flag:
        d.save(gv.plots_path + 'flowchart_HETDEX_subsets.pdf')

# Stripe 82
with schemdraw.Drawing(fontsize=11) as d:
    d += (HETDEX := flow.Start(w=3, h=1.5).label('Stripe 82\n369,093'))
    d += schemdraw.elements.lines.Gap().at(HETDEX.S)

    d += (Labelled := flow.RoundBox(w=3, h=1.5, anchor='ENE').label('Labelled\n3,304'))
    d += (Unlabelled := flow.RoundBox(w=3, h=1.5, anchor='WNW').label('Unlabelled\n365,789'))
    d += flow.Arrow().length(d.unit/3).at(HETDEX.S).to(Labelled.N)
    d += flow.Arrow().length(d.unit/3).at(HETDEX.S).to(Unlabelled.N)
    d += schemdraw.elements.lines.Gap().at(Labelled.S)

    # d.draw()
    if save_plot_flag:
        d.save(gv.plots_path + 'flowchart_S82_subsets.pdf')