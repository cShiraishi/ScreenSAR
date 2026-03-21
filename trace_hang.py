import sys
import tracer
import trace
tracer = trace.Trace(count=False, trace=True, ignoredirs=[sys.prefix, sys.exec_prefix])
tracer.run('import src.ui.auth')
