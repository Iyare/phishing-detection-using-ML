import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt
from urllib3.exceptions import LocationParseError


st.title("Phishing Detection Project")

st.write("This is a Content-Based ML-Based app is developed as a project work.")

with st.expander("PROJECT DETAILS"):
    st.subheader("Approach")
    st.write("This project adopted a supervised learning method in training several Scikit-learn models commonly used in classification projects. These models include: Gaussian Naive Bayes, Support Vector Machines, Decision Trees, Random Forest, AdaBoost, Neural Network, and K-Neighbours. After training the models with labeled data, models were tested with a seperate test data. Results detailing the accuracy, recall and precision values were obtained for each model.  The best performing model was the Random Forest with a precision of 81%, recall of 69%, and a 98% accuracy score This web application allows users to use the trained models to detect  for  nto phishing or legitimate based on their HTML content NOT URL features like URL length, etc.")
    
    st.write("The request library and BeautifulSoup4 module in python was used to scrap the webpages, parse and extract features")
    st.write("The source code and datasets are available in the below Github")
    st.write("_https://www.github.com/_")
    
    st.subheader("Data Sources")
    st.write("Phishing URLs were obtained from _'phishtank.org'_ while legitimate URLs were downloaded from _'tranco-list.eu'_ .")
    st.write("A total of **_26584_** feature vectors were extracted from both phishing and legitimate URLs downloaded.**_16060_** - were extracted from legitimate webpages while **_10524_** were extracted from phishing websites.")
    
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
    st.write("The dataset used in training and testing the models was created by defining and extracting specific features. These features were chosen based on several literatures reviewed and manual inspection of some phishing and legitimate websites")
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
    st.write('https://authmycookie.com/rt4.php?r3=CRA6RBEOQxAMFkFdRQlJF15bDhUCQQtSDkEDAAkOGVcNBQkBQB8%3D')
    st.write('https://auth--m-start--ttrezr.webflow.io/')
    st.write('http://evri.poaekhgroup.xyz')
    st.caption('Please note that phishing URLs have a very short lifecycle. So the above URLs might be offline at anytime!')
    st.caption("Visit _https://www.phishtank.org_ for newly listed phishing URLs.")

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

