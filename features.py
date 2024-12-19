from bs4 import BeautifulSoup
 
with open("dataset/10.html", encoding="utf-8") as f:
    file = f.read()
    
soup = BeautifulSoup(file, "html.parser")

# feature list
# Has title?
def has_title(soup):
    if soup.find_all("title"):
        return 1
    else:
        return 0
    
# Has Input?
def has_input(soup):
    if soup.find_all("input"):
        return 1
    else:
        return 0

# has button?
def has_button(soup):
    if soup.find_all("button"):
        return 1
    else:
        return 0
    
# has image?
def has_image(soup):
    if soup.find_all("img"):
        return 1
    else:
        return 0
    
# has submit button or input
def has_submit(soup):
    if soup.find_all("button", type="submit") or soup.find_all("input", type="submit"):
        return 1
    else:
        return 0

# has links
def has_link(soup):
    if soup.find_all("a"):
        return 1
    else:
        return 0

# has password
def has_password(soup):
    if  soup.find_all("input", type="password") or soup.find_all("input", id="password") or soup.find_all("input", attrs={"name":"password"}):
        return 1
    else:
        return 0

# has email input
def has_email_input(soup):
    if  soup.find_all("input", type="email") or soup.find_all("input", attrs={"name":"email"}) or soup.find_all("input", id="email"):
        return 1
    else:
        return 0

# has hidden element
def has_hidden_element(soup):
    for input in soup.find_all("input"):
        if input.get("type") == "hidden":
            return 1
        else:
            pass
    return 0

# has an audio
def has_audio(soup):
    if soup.find_all("audio"):
        return 1
    else:
        return 0

# has a video
def has_video(soup):
    if soup.find_all("video"):
        return 1
    else:
        return 0

#-------------
#has h1
def has_h1(soup):
    if soup.find_all("h1"):
        return 1
    else:
        return 0
    
#has h2
def has_h2(soup):
    if soup.find_all("h2"):
        return 1
    else:
        return 0
#has h3
def has_h3(soup):
    if soup.find_all("h3"):
        return 1
    else:
        return 0
    
#has footer
def has_footer(soup):
    if soup.find_all("footer"):
        return 1
    else:
        return 0
    
#has form
def has_form(soup):
    if soup.find_all("form"):
        return 1
    else:
        return 0
       
#has text area
def has_textarea(soup):
    if soup.find_all("textarea"):
        return 1
    else:
        return 0

#has nav
def has_nav(soup):
    if soup.find_all("nav"):
        return 1
    else:
        return 0
    

#has iframe
def has_iframe(soup):
    if soup.find_all("iframe"):
        return 1
    else:
        return 0
    
#has object
def has_object(soup):
    if soup.find_all("object"):
        return 1
    else:
        return 0
    
#has tables
def has_table(soup):
    if soup.find_all("table"):
        return 1
    else:
        return 0
    
#has picture tags
def has_picture(soup):
    if soup.find_all("picture"):
        return 1
    else:
        return 0
    
def has_text_input(soup):
    for input in soup.find_all("input"):
        if input.get("type") == "text":
            return 1
    return 0

#num of divs
def num_divs(soup):
    return len(soup.find_all("div"))

#num of meta
def num_metas(soup):
    return len(soup.find_all("meta"))

#number of figure
def num_figures(soup):
    return len(soup.find_all("figure"))

#num of tables
def num_tables(soup):
    return len(soup.find_all("table"))

#number of spans
def num_spans(soup):
    return len(soup.find_all("span"))

#number of anchor elements
def num_anchors(soup):
    return len(soup.find_all("a"))

'''#number of images with alt
def num_alts(soup):
    altsList = []
    for attr in soup.find_all("img"):
        if attr.get("alt") != None:
            altsList.append(attr.get("alt"))
        else:
            pass
    return len(altsList)
'''
# number of inputs
def num_inputs(soup):
    return len(soup.find_all("input"))
    

# number of buttons
def num_buttons(soup):
    return len(soup.find_all("button"))

# number of images
def num_images(soup):
    return len(soup.find_all("img"))


# number of options
def num_options(soup):
    return len(soup.find_all("option"))

# number of list
def num_lists(soup):
    return len(soup.find_all("li"))

# number of Table Headings
def num_th(soup):
    return len(soup.find_all("th"))

# number of Table rows
def num_tr(soup):
    return len(soup.find_all("tr"))

# number of hrefs
def num_hrefs(soup):
    thisList = []
    for link in soup.find_all("a"):
        if link.get("href") != None:
            thisList.append(link.get("href"))
        else:
            pass
    return len(thisList)

# number of paragraphs
def num_paragraphs(soup):
    return len(soup.find_all("p"))

# number of source tags
def num_sources(soup):
    return len(soup.find_all("source"))

# number of scripts
def num_scripts(soup):
    return len(soup.find_all("script"))

# length of title
def title_length(soup):
    if soup.find_all("title"):
        return len(soup.title.get_text())
    else: 
        return 0

# length of text on page
def text_length(soup):
    content = soup.get_text()
    return len(content)



