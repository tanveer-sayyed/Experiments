from argparse import ArgumentParser, RawTextHelpFormatter

if __name__ == '__main__':
    my_parser = ArgumentParser(prog="argument_parser.py",
                               description="python argument_parser.py A 1.1",
                               epilog="Enjoy the program! :)\n",
                               formatter_class=RawTextHelpFormatter)
    my_parser.add_argument('_string',
                           type=str,
                           metavar='_string',                    
                           help="string variable")
    my_parser.add_argument('_float',
                           type=float,
                           metavar='_float',                           
                           help="float variable")
    args = my_parser.parse_args()
    print(args._string)
    print(args._float)
    