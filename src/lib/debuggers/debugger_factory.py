from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .trictdet import TriCtdetDebugger
from .dbctdet import DbCtdetDebugger

debugger_factory = {
  'trictdet': TriCtdetDebugger,
  'dbctdet': DbCtdetDebugger,
}
