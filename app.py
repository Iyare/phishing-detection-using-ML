import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt
from urllib3.exceptions import LocationParseError
import random as random
from random import choice


st.title("Phishing Detection App - Project")

st.write("This is a Content-Based ML-Based app is developed as a project work.")

with st.expander("PROJECT DETAILS"):
    st.subheader("Approach")
    st.write("I used supervised learning to classify phishing and legitimate websites"
             "Sckit-learn  was also used for the ML Models.")
    st.write("For  this project, I created my own dataset and defined features based on some reviewed literature and manual inspection of some phishing websites")
    st.write("The request library was used to scrap the webpages and BeautifulSoup module  was used to parse and extract features")
    st.write("The  source code and datasets are available in the below  Github")
    st.write("_https://www.github.com/_")
    
    st.subheader("Datasets")
    st.write(" I used _'phishtank.org'_ & _'tranco-list.eu'_ as data sources.")
    st.write("Dataset contained a total of **_26584_** websites ===> **_16060_ - legitimate** websites | **_10524_** phishing websites")
    
    # ----- FOR THE PIE CHART ----- #
    labels = 'phishing', 'legitimate'
    phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    legitimate_rate = 100 - phishing_rate
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)
    # ----- !!!!! ----- #
    
    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))


    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(ml.df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )

    st.subheader('Features')
    st.write('I used only content-based features. I didn\'t use url-based faetures like length of url etc.'
             'Most of the features extracted using find_all() method of BeautifulSoup module after parsing html.')

    st.subheader('Results')
    st.write('I used 7 different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.'
             'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
             'Comparison table is below:')
    st.table(ml.df_results)
    st.write('NB --> Gaussian Naive Bayes')
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('RF --> Random Forest')
    st.write('AB --> AdaBoost')
    st.write('NN --> Neural Network')
    st.write('KN --> K-Neighbours')

with st.expander('SOME PHISHING URLs:'):
    st.write('https://krajanelogin.webflow.io')
    st.write('https://auth--m-start--ttrezr.webflow.io/')
    st.write('http://evri.poaekhgroup.xyz')
    st.caption('Please note that phishing URLs have a very short lifecycle. So the above URLs might be offline at anytime!')
    st.caption("Visit _https://www.phishtank.org_ for newly listed phishing URLs for testing")

choice = st.selectbox("Please select your machine learning model",
                 [
                     'Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
                     'AdaBoost', 'Neural Network', 'K-Neighbours'
                 ]
                )

model = ml.nb_model

if choice == 'Gaussian Naive Bayes':
    model = ml.nb_model
    st.write('GNB model is selected!')
elif choice == 'Support Vector Machine':
    model = ml.svm_model
    st.write('SVM model is selected!')
elif choice == 'Decision Tree':
    model = ml.dt_model
    st.write('DT model is selected!')
elif choice == 'Random Forest':
    model = ml.rf_model
    st.write('RF model is selected!')
elif choice == 'AdaBoost':
    model = ml.ab_model
    st.write('AB model is selected!')
elif choice == 'Neural Network':
    model = ml.nn_model
    st.write('NN model is selected!')
else:
    model = ml.kn_model
    st.write('KN model is selected!')


url = st.text_input('Enter the URL in full. Example: https://example.com')
# check the url is valid or not
if st.button('Check URL'):
    with st.spinner("Please wait..."):
        try:
                     
            headers = {
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "max-age=0",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
                 }
            
            response = re.get(url, headers = headers, verify = False, timeout=30)
            if response.status_code != 200:
                print("HTTP connection was not successful for the URL: ", url)
                st.error(f"HTTP connection was not successful for this URL. Error code: {response.status_code}")
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = [fe.create_vector(soup)]  # it should be 2d array, so I added []
                result = model.predict(vector)
                if result[0] == 0:
                    st.success("This web page seems a legitimate!")
                    st.balloons()
                else:
                    st.warning("Attention! This is a potential PHISHING site!")
                    # st.snow()
                    
        except LocationParseError as e: 
            print("URL:An error occurred while parsing the location:", e)
            st.error("Sorry, something went wrong with the connection", icon="🚨")
            st.exception(e)
                        
        except re.exceptions.RequestException as e:
            print("--> ", e)
            st.error("Sorry, something went wrong with the connection", icon="🚨")
            st.exception(e)

