from argparse import ArgumentParser, RawTextHelpFormatter

from my_logger import log_this_error
from utils import read_xml, process_xml, write_csv, s3_upload

if __name__ == "__main__":
    try:
        my_parser = ArgumentParser(
            prog="xml_to_csv_converter.py",
            description="python xml_to_csv_converter.py xml_file_path",
            epilog="Enjoy the program! :)\n",
            formatter_class=RawTextHelpFormatter)
        my_parser.add_argument(
            'xml_file_path',
            type=str,
            metavar='p',
            help="string variable")
        # convert
        write_csv(
            process_xml(
                read_xml(my_parser.parse_args().xml_file_path)
                )
            )
        # push to bucket
        print("...file uploaded to S3 bucket:",
              s3_upload(file_name = "output.csv", bucket = "s3_bucket"))
        print("Done!")
    except Exception as e:
        log_this_error(e)
