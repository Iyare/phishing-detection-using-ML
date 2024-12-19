#This file extractes features from webpages using the features.py file and organizes the data into dataframe

from bs4 import BeautifulSoup
import os
import features as fe
import pandas as pandas

file_name = "dataset/10.html"

#1 Function that opens a HTML file and returns the content
def open_file(file_name):
    with open(file_name, encoding="utf-8") as f:
        return f.read()


#2 Function that creates a beautiful soup object
def create_soup(file_content):
    soup = BeautifulSoup(file_content, "html.parser")
    return soup

#3 Function that creates a vector by running all feature functions for the soup 
def create_vector(soup):
    return [
        fe.has_title(soup),
        fe.has_input(soup),
        fe.has_button(soup),
        fe.has_image(soup),
        fe.has_submit(soup),
        fe.has_link(soup),
        fe.has_password(soup),
        fe.has_email_input(soup),
        fe.has_hidden_element(soup),
        fe.has_audio(soup),
        fe.has_video(soup),
        fe.has_h1(soup),
        fe.has_h2(soup),
        fe.has_h3(soup),
        fe.has_footer(soup),
        fe.has_form(soup),
        fe.has_textarea(soup),
        fe.has_nav(soup),
        fe.has_iframe(soup),
        fe.has_object(soup),
        fe.has_picture(soup),
        fe.has_text_input(soup),
        fe.num_inputs(soup),        
        fe.num_buttons(soup),
        fe.num_options(soup),
        fe.num_lists(soup),
        fe.num_th(soup),
        fe.num_tr(soup),
        fe.num_hrefs(soup),
        fe.num_paragraphs(soup),
        fe.num_scripts(soup),
        fe.title_length(soup),
        fe.text_length(soup),
        fe.num_anchors(soup),
        fe.num_images(soup),
        fe.num_divs(soup),
        fe.num_figures(soup),
        fe.num_metas(soup),
        fe.num_sources(soup),
        fe.num_spans(soup),
        fe.num_tables(soup)       
    ]


#4 Runs step 1,2,3 for all HTML files and create a 2-D array
folder = "dataset"

def create_2d_list(folder_name):
    directory = os.path.join(os.getcwd(), folder_name)
    data = []
    for file in sorted(os.listdir(directory)):
        soup = create_soup(open_file(directory + "/" + file))
        data.append(create_vector(soup))
    return data

#5 Creates a Dataframe using 2-D Array
data = create_2d_list(folder)

columns = [
        "",
        "has_title",
        "has_input",
        "has_button",
        "has_image",
        "has_submit",
        "has_link",
        "has_password",
        "has_email_input",
        "has_hidden_element",
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
        "has_object"
        "has_picture",
        "has_text_input",
        "num_inputs",
        "num_buttons",
        "num_options",
        "num_lists",
        "num_th",
        "num_tr",
        "num_hrefs",
        "num_paragraphs",
        "num_scripts",
        "title_length",
        "text_length",
        "num_anchors",
        "num_images",
        "num_divs",
        "num_figures",
        "num_metas",
        "num_sources",
        "num_spans", 
        "num_tables",
        
]

dataframe = pandas.DataFrame(data=data, columns=columns)

# print(dataframe)
