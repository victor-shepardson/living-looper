import sys
import fire

from livinglooper.export import main as export

def help():
    print("""
    available subcommands:
        export: export a Living Looper torchscript model
    """)

def _main():
    try:
        if sys.argv[1] == 'export':
            sys.argv = sys.argv[1:]
            fire.Fire(export)
        else:
            help()
    except IndexError:
        help()

if __name__=='__main__':
    _main()