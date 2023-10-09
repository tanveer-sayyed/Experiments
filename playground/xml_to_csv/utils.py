import csv
import xml.dom.minidom
from boto3 import client

def read_xml(xml_file_path):
    """

    Parameters
    ----------
    xml_file_path : str

    Returns
    -------
    dom : xml.dom.minidom.Document

    """
    print(f"...reading xml file {xml_file_path}")
    return xml.dom.minidom.parse(xml_file_path)

def process_xml(dom):
    """
    Parameters
    ----------
    dom : xml.dom.minidom.Document

    Returns
    -------
    file : [[str]]

    """
    print("...processing the xml file")
    pretty_xml = dom.toprettyxml()
    pretty_xml = pretty_xml.split("\n")
    start, end = [], []
    for idx in range(len(pretty_xml)):
        if "FinInstrmGnlAttrbts" in pretty_xml[idx]:
            if "/" not in pretty_xml[idx]:
                start.append(idx)
            else:
                end.append(idx)
    
    assert(len(start) == len(end))
    
    file = [["FinInstrmGnlAttrbts.Id", 
             "FinInstrmGnlAttrbts.FullNm",
             "FinInstrmGnlAttrbts.ShrtNm",
             "FinInstrmGnlAttrbts.ClssfctnTp",
             "FinInstrmGnlAttrbts.CmmdtyDerivInd",
             "FinInstrmGnlAttrbts.NtnlCcy",
             "Issr"]]
    
    print("...breaking down xml components")
    for idx in range(len(start)):
        f = []
        for line in pretty_xml[start[idx] + 1 : end[idx]]:
            a, b = line.strip("\t").split("</"); f.append(a[len(b) + 1:])
        a, b = pretty_xml[end[idx] + 1].strip("\t").split("</")
        f.append(a[len(b) + 1:]); file.append(f)
    return file

def write_csv(file):
    """
    Parameters
    ----------
    file : [[str]]

    Returns
    -------
    None.

    """
    print("...writing to 'output.csv'")
    with open('output.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile,
                                delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
        for line in file:
            spamwriter.writerow(line)

def s3_upload(file_name, bucket, object_name=None):
    """

    Parameters
    ----------
        file_name: File to upload
        bucket: Bucket to upload to
        object_name: S3 object name. If not specified then file_name is used
        
    Returns:
        True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    s3_client = client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except:
        return False
    return True
