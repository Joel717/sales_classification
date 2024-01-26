#importing necessary packages
import streamlit as st
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import pickle 
import joblib
import pandas as pd 
from PIL import Image
#setting the page configuration
st.set_page_config(layout="wide")
st.markdown("""
<div style='text-align:center; font-family:"Times New Roman";'>
    <h1 style='color:#0000ff;'>Sales Conversion Classifier</h1>
</div>
""", unsafe_allow_html=True)
#setting a background for the page
def setting_bg():
    local_image_path = "https://imgur.com/TouAJ16.png"
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: url("{local_image_path}");
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

setting_bg()
#loading the mapped dictionaries for label encoder
with open('mapping_dict1.pkl', 'rb') as file:
    mapping_dict_channelGrouping = pickle.load(file)

with open('mapping_dict2.pkl', 'rb') as file:
    mapping_dict_device_browser = pickle.load(file)

with open('mapping_dict3.pkl', 'rb') as file:
    mapping_dict_device_operatingSystem = pickle.load(file)

with open('mapping_dict4.pkl', 'rb') as file:
    mapping_dict_Products = pickle.load(file)

with open('mapping_dict5.pkl', 'rb') as file:
    mapping_dict_region = pickle.load(file)
#loading csv for visualizations 
dfi=pd.read_csv('dff.csv')
x_test=pd.read_csv('x_test.csv')
y_test=pd.read_csv('y_test.csv')
#loading scaler and model
scaler = joblib.load('standard_scaler.pkl')

model = joblib.load('dtree.pkl')
model1 = joblib.load('svm_model.pkl')
model2 = joblib.load('random_forest_model.pkl')
#creating tabs 
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Home', 'Predict', 'EDA', 'Models', 'About'])

#setting tab and dropdown for feature selection
with tab2:
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap='small')

    with col1:
        count_session = st.text_input("Enter Session Count")
    with col2:
        count_hit = st.text_input("Enter hit Count")
    with col3:
        num_interactions = st.text_input("Enter number of interactions")
    with col4:
        channelGrouping = st.selectbox('Channel Grouping', options=list(mapping_dict_channelGrouping.keys()))
    
    col5, col6, col7, col8 = st.columns([1, 1, 1, 1], gap='small')

    with col5:
        device_browser = st.selectbox('Device Browser', options=list(mapping_dict_device_browser.keys()))
    with col6:
        device_operatingSystem = st.selectbox('Operating System', options=list(mapping_dict_device_operatingSystem.keys()))
    with col7:
       Products = st.selectbox('Select Products', options=list(mapping_dict_Products.keys()))
    with col8:
        region = st.selectbox('Select Region', options=list(mapping_dict_region.keys()))

# Create DataFrame after user inputs
    df = pd.DataFrame({
        'count_session': [float(count_session)],
        'count_hit': [float(count_hit)],
        'num_interactions': [float(num_interactions)],
        'channelGrouping': [channelGrouping],
        'device_browser': [device_browser],
        'device_operatingSystem': [device_operatingSystem],
        'Products': [Products],
        'region': [region]})
#creating a dataframe with mapped features 
    df_encoded = df.copy()
    df_encoded['channelGrouping'] = df['channelGrouping'].map(mapping_dict_channelGrouping)
    df_encoded['device_browser'] = df['device_browser'].map(mapping_dict_device_browser)
    df_encoded['device_operatingSystem'] = df['device_operatingSystem'].map(mapping_dict_device_operatingSystem)
    df_encoded['Products'] = df['Products'].map(mapping_dict_Products)
    df_encoded['region'] = df['region'].map(mapping_dict_region)

    dfc=df_encoded.drop(['count_session','count_hit','num_interactions'],axis=1)
# Select numeric columns for scaling
    dfn = df_encoded.drop(['channelGrouping','device_browser','device_operatingSystem','Products','region'],axis=1)

# Scale numeric columns
    dfn_scaled = pd.DataFrame(scaler.transform(dfn), columns=dfn.columns)



    dff=pd.concat([dfn_scaled,dfc],axis=1)

    if st.button('Predict'):

        prediction=model.predict(dff)

        if prediction==1:
            st.success('Sale Converted')
        else:
            st.warning('Sale Not Converted')
#creating a about section in tab3
with tab5:
     with st.container():
                    st.markdown(
                '''
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;font-family: "Times New Roman">
                    <p style="font-size: 18px;">About the creator:</p>
                    <p style="font-size: 18px;">The app is created by: Joel Gracelin</p>
                    <p style="font-size: 18px;">This app is created as a part of Guvi Master Data Science course</p>
                    <p style="font-size: 18px;">Domain: Sales and Ecommerce</p>
                    <p style="font-size: 18px;">Inspired by Conversion classification models</p>
                    <p style="font-size: 18px;">[Ghithub](https://github.com/Joel717)</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
#creating a home tab with info about the model and project 
with tab1:
    col1,col2 =st.columns([1,1],gap='small')
    with col1:
     with st.container():
         
        st.markdown(
        '''<div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
            <p style="font-size: 18px;">A sales conversion refers to the process of turning a potential customer (or lead) into an actual paying customer.
            <p style="font-size: 18px;">It is a critical metric in sales and marketing, indicating the successful completion of a desired action, such as making a purchase, signing up for a service,
            or taking any other desired action that aligns with the business's goals.</p>
            <p style="font-size: 18px;">The conversion process typically involves moving a prospect through the sales funnel, which consists of various stages,
            such as awareness, interest, consideration, and finally, the decision to make a purchase.</p>
            <p style="font-size: 18px;">Successful conversions depend on effective communication, persuasion, and addressing the customer's needs and concerns.
            </div>
            ''',unsafe_allow_html=True)
    with col2:
        img_path = 'C:\\Users\\devli\\fp\\Customers-Sales-Funnel.jpg'
        img = Image.open(img_path)
        img_width = 400
        st.image(img, width=img_width)
    
    col3,col4 =st.columns([1,1],gap='small')

    with col3:
        img_path = 'C:\\Users\\devli\\fp\\Machine-learning-def-.png'
        img = Image.open(img_path)
        img_width = 500
        st.image(img, width=img_width)

    with col4:
        with st.container():
         
            st.markdown(
                '''
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
                    <p style="font-size: 18px;">Welcome to the Sales Conversion Classifier!</p>
                    <p style="font-size: 18px;">This tool leverages machine learning to predict whether a lead will convert into a paying customer.</p>
                    <p style="font-size: 18px;">Key Features:</p>
                    <ul style="font-size: 18px;">
                        <li>Utilizes a Decision Tree model for classification.</li>
                        <li>Input key features such as session count, hit count, and more to make predictions.</li>
                        <li>Intelligently encodes categorical features like channel grouping, device browser, etc.</li>
                        <li>Scalable and adaptable for real-time predictions.</li>
                    </ul>
                </div>
                ''',
                unsafe_allow_html=True
            )
with tab3:
    #visualizations for the eda analysis
    st.markdown('EDA of the data are given below')
    col1,col2=st.columns([1,1],gap='small')
    #pairplot
    with col1:
        st.title('Pair plot')
        fig=sns.pairplot(dfi, hue="has_converted")
        
        st.pyplot(fig,use_container_width=True)
    #heatmap    
    with col2:
        st.title('Correlation Heatmap')
        import plotly.express as px
        corrmat=dfi.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corrmat, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        
    col3,col4=st.columns([1,1],gap='small')
    #boxplot
    with col3:
        st.title('Box Plot')
        st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable matplotlib warning

        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 12))

        # List of variables for which you want to create box plots
        variables = ['count_session', 'count_hit', 'num_interactions', 'channelGrouping',
                    'device_browser', 'device_operatingSystem', 'Products', 'region']

        # Create a collage of box plots
        for i, variable in enumerate(variables, start=1):
            plt.subplot(4, 2, i)
            sns.boxplot(x='has_converted', y=variable, data=dfi)
            plt.title(f'Box Plot for {variable}')
            plt.xlabel('Has Converted')
            plt.ylabel(variable)

        # Adjust layout
        plt.tight_layout()

        # Display the box plots using st.pyplot
        st.pyplot(plt.gcf(), use_container_width=True)
    #3d scatter plot
    with col4:
        from mpl_toolkits.mplot3d import Axes3D
        st.title('3D Scatter Plot')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(dfi['count_session'], dfi['count_hit'], dfi['num_interactions'])

        # Customize the plot if needed
        ax.set_xlabel('Count Session')
        ax.set_ylabel('Count Hit')
        ax.set_zlabel('Num Interactions')
        ax.set_title('3D Scatter Plot')

        # Display the plot in the Streamlit app
        st.pyplot(fig, use_container_width=True)

with tab4:
    #using different models
    col1,col2,col3=st.columns([1,1,1],gap='small')
    with col1:
        st.title('SVM Model')
        with st.container():
                st.markdown(
                    '''
                    <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
                        <p style="font-size: 18px;">Support Vector Machine (SVM) Model Overview:</p>
                        <p style="font-size: 18px;">A Support Vector Machine is a powerful supervised machine learning algorithm used for both classification and regression tasks.</p>
                        <p style="font-size: 18px;">Key Features:</p>
                        <ul style="font-size: 18px;">
                            <li>Effective for both linear and non-linear data separation.</li>
                            <li>Maximizes the margin between classes for robust generalization.</li>
                            <li>Works well in high-dimensional spaces.</li>
                            <li>Kernel trick allows mapping input features into higher-dimensional space.</li>
                            <li>Sensitive to feature scaling.</li>
                        </ul>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
     
    

    with col2:
     st.title('Random Forest Model')
     with st.container():
            st.markdown(
                '''
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
                    <p style="font-size: 18px;">Random Forest Model Overview:</p>
                    <p style="font-size: 18px;">Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of individual trees.</p>
                    <p style="font-size: 18px;">Key Features:</p>
                    <ul style="font-size: 18px;">
                        <li>Ensemble of decision trees for improved accuracy and generalization.</li>
                        <li>Handles missing values and maintains accuracy even with a large number of features.</li>
                        <li>Reduces overfitting compared to individual decision trees.</li>
                        <li>Can handle both categorical and numerical data.</li>
                        <li>Random feature selection and bootstrapped sampling enhance diversity.</li>
                    </ul>
                </div>
                ''',
                unsafe_allow_html=True
            )
    
    with col3:
        st.title('Decision Tree Model')
        with st.container():
            st.markdown(
                '''
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
                    <p style="font-size: 18px;">Decision Tree Model Overview:</p>
                    <p style="font-size: 18px;">A Decision Tree is a popular machine learning algorithm used for both classification and regression tasks. It builds a tree-like structure by recursively splitting the dataset based on the most significant features.</p>
                    <p style="font-size: 18px;">Key Features:</p>
                    <ul style="font-size: 18px;">
                        <li>Simple yet powerful for interpreting and visualizing decisions.</li>
                        <li>Handles both numerical and categorical data.</li>
                        <li>Non-parametric and requires minimal data preprocessing.</li>
                        <li>Prone to overfitting, but techniques like pruning help mitigate it.</li>
                        <li>Efficient for binary and multi-class classification problems.</li>
                    </ul>
                </div>
                ''',
                unsafe_allow_html=True
            )
        
        
    col1,col2,col3=st.columns([1,1,1],gap='small')       
    with col1:
        #using svm model
     if st.button('Predict (SVM model)'):

        prediction1=model1.predict(dff)

        if prediction1==1:
            st.success('Sale Converted')
        else:
            st.warning('Sale Not Converted')
    with col2:
        #using random forest model
     if st.button('Predict (Random Forest model)'):

        prediction2=model2.predict(dff)

        if prediction2==1:
            st.success('Sale Converted')
        else:
            st.warning('Sale Not Converted')
    with col3:
        #using dtree model
        if st.button('Predict (Decision Tree model)'):

            prediction=model.predict(dff)

            if prediction==1:
                st.success('Sale Converted')
            else:
                st.warning('Sale Not Converted')
    col1,col2,col3=st.columns([1,1,1],gap='small') 
    with col1:
        with st.container():
            st.markdown(
                '''
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
                    <p style="font-size: 18px;">Support Vector Machine (SVM) Model Performance Metrics:</p>
                    <p style="font-size: 18px;">The following metrics provide insights into the effectiveness of the SVM model:</p>
                    <ul style="font-size: 18px;">
                        <li>F1 Score: 0.7655</li>
                        <li>Accuracy: 0.7731</li>
                        <li>Precision: 0.7838</li>
                        <li>Recall: 0.7731</li>
                    </ul>
                    <p style="font-size: 18px;">These metrics illustrate the SVM model's performance in terms of precision, recall, and accuracy in classifying instances.</p>
                </div>
                ''',
                unsafe_allow_html=True
            )

    with col2:
        with st.container():
            st.markdown(
                '''
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
                    <p style="font-size: 18px;">Random Forest Model Performance Metrics:</p>
                    <p style="font-size: 18px;">The following metrics provide insights into the effectiveness of the Random Forest model:</p>
                    <ul style="font-size: 18px;">
                        <li>F1 Score: 0.8094</li>
                        <li>Accuracy: 0.8105</li>
                        <li>Precision: 0.8104</li>
                        <li>Recall: 0.8105</li>
                    </ul>
                    <p style="font-size: 18px;">These metrics demonstrate the Random Forest model's ability to balance precision, recall, and accuracy in classifying instances.</p>
                </div>
                ''',
                unsafe_allow_html=True
            )

    with col3:
        with st.container():
            st.markdown(
                '''
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
                    <p style="font-size: 18px;">Decision Tree Model Performance Metrics:</p>
                    <p style="font-size: 18px;">The following metrics provide insights into the effectiveness of the Decision Tree model:</p>
                    <ul style="font-size: 18px;">
                        <li>F1 Score: 0.8216</li>
                        <li>Accuracy: 0.8225</li>
                        <li>Precision: 0.8222</li>
                        <li>Recall: 0.8225</li>
                    </ul>
                    <p style="font-size: 18px;">These metrics indicate the Decision Tree model's ability to balance precision, recall, and accuracy in classifying instances.</p>
                </div>
                ''',
                unsafe_allow_html=True
            )

    


