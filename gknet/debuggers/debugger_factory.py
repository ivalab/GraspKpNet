



from .trictdet import TriCtdetDebugger
from .dbctdet import DbCtdetDebugger

debugger_factory = {
  'trictdet': TriCtdetDebugger,
  'dbctdet': DbCtdetDebugger,
}
