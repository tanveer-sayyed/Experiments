import unittest
import xml.dom.minidom

from xml_to_csv_converter import process_xml

class xml_to_csv_converter_test(unittest.TestCase):

    def test_process_xml(self):
        input_xml_string = "<?xml version='1.0' encoding='ISO-8859-1'?>" +\
        "<body><FinInstrmGnlAttrbts><Id>ID</Id>" +\
        "<FullNm>FULLNM</FullNm><ShrtNm>SHRTNM</ShrtNm>" +\
        "<ClssfctnTp>CLSSFCTNTP</ClssfctnTp>" +\
        "<CmmdtyDerivInd>CMMDTYDERIVIND</CmmdtyDerivInd>" +\
        "<NtnlCcy>NTNLCCY</NtnlCcy></FinInstrmGnlAttrbts>" +\
        "<Issr>ISSR</Issr>" +\
        "</body>"
        desired_output = [['FinInstrmGnlAttrbts.Id',
                           'FinInstrmGnlAttrbts.FullNm',
                           'FinInstrmGnlAttrbts.ShrtNm',
                           'FinInstrmGnlAttrbts.ClssfctnTp',
                           'FinInstrmGnlAttrbts.CmmdtyDerivInd',
                           'FinInstrmGnlAttrbts.NtnlCcy',
                           'Issr'],
                         ['ID', 'FULLNM', 'SHRTNM', 'CLSSFCTNTP',
                          'CMMDTYDERIVIND', 'NTNLCCY', 'ISSR']]
        dom = xml.dom.minidom.parseString(input_xml_string)
        output = process_xml(dom)
        self.assertEqual(output, desired_output)

if __name__ == '__main__':
    unittest.main()