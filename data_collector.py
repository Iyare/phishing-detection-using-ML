#data_collection from csv file and processing

import requests as req
from bs4 import BeautifulSoup
import pandas as pd
import feature_extraction as fe
from urllib3.exceptions import LocationParseError


req.packages.urllib3.disable_warnings()
session = req.Session()
session.verify = False


# Import CSV to dataframe
url_file_name = "phishing_urls.csv"

data_frame = pd.read_csv(url_file_name)


#retrieve only "url" column and convert it to a list
url_list= data_frame["url"].to_list()


#Due to the size of the phishing url list and the verified tranco list. A restriction is  placed  on the number of urls to be taken at a go.#


#Restrict the URL count

begin = 0
end = 25000
collection_list = url_list[begin:end]

#Adding "http" only for only legitimated URLS/domains
tag = "http://"
collection_list = [tag + url for url in collection_list]

#function to scrape the content of the URL and convert to a structured form for each URL

def create_structured_data(url_list):
    data_list = []
    for i in range(0, len(url_list)):
        try:
            response = req.get(url_list[i], verify=False, timeout=30)
            if response.status_code != 200:
                print (i, ". HTTP connection failed for this URL", url_list[i])
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = fe.create_vector(soup)
                vector.append(str(url_list[i]))
                data_list.append(vector)
        except LocationParseError as e: 
            print(i,"An error occurred while parsing the location:", e)
            continue
        except req.exceptions.RequestException as e:
            print(i,"---->", e)
            continue
        
    return data_list

data = create_structured_data(collection_list)


columns = [
        "has_title",
        "has_input",
        "has_button",
        "has_image",
        "has_submit",
        "has_link",
        "has_password",
        "has_email",
        "has_hidden_input",
        "has_audio",
        "has_video",
        "has_h1",
        "has_h2",
        "has_h3",
        "has_footer",
        "has_form",
        "has_textarea",
        "has_nav",
        "has_iframe",
        "has_table",
        "has_picture",
        "num_divs",
        "num_metas",
        "num_figures",
        "num_tables",
        "num_spans",
        "num_anchors",
        "num_alts",
        "num_inputs",
        "num_buttons",
        "num_images",
        "num_options",
        "num_lists",
        "num_th",
        "num_tr",
        "num_href",
        "num_paragraphs",
        "num_scripts",
        "num_titles",
        "text_length",
        "url",
]

#creating dataframe

df = pd.DataFrame(data=data, columns=columns)

#labelling 0 for good websites, 1 for phishing websites
df["label"] = 1


#Because of the restriction a new csv file to which new records are appended for each run cycle

df.to_csv("structured_data_phishing.csv", mode="a", index=False) #header should be true for the first run only 