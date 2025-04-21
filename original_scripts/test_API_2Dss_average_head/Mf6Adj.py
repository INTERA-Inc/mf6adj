import os
import sys

currentdirectory = os.path.abspath(sys.argv[0]+"/..")
print(currentdirectory+"\\%s"%"GWF_2D_SS.py")


class MF6Adj:

    def solve_forward(self, name):
        self.name = name
        return %run name