"""
schema_analysis.tube — self-contained tube-tilt analysis package.

    from schema_analysis.tube import load
    from schema_analysis.tube import treatments
    from schema_analysis.tube.plots import sensitivity_heatmap

    s = load(clean=True).balance()
    s = s.select_trials('valid == True')
    s.print_summary()
    s.plots.plot_d(save='figure_d_bars.png')
"""

from .sessions import load, TubeSessions, Session
from . import treatments

__all__ = ['load', 'TubeSessions', 'Session', 'treatments']
