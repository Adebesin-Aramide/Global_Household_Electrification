import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from streamlit_option_menu import option_menu
from PIL import Image  

# Set page configuration
st.set_page_config(
    page_title='Global Household Electrification',
    page_icon='Hamoye_Capstone.PNG',
    layout='wide'
)

col1, col2 = st.columns((1, 1))

with col1:
    logo = Image.open('Hamoye_Capstone.PNG')
    st.image(logo, width=430)  # Adjust the width as needed

with col2:
    st.title('     Global Household Electrification')  # Remove color formatting
    st.subheader("    _Team GCP, Hamoye HDSC Spring '23 Capstone Project_")

with st.sidebar:
    selected = option_menu(menu_title='Main menu',options=['Home','Visualizations','Prediction','Team'], 
    icons=['house-fill','bar-chart-fill','globe','x-diamond-fill','person-fill'],
    menu_icon="cast", default_index=0,)

#with st.sidebar:
 #   selected = option_menu(menu_title='Main menu', ['Home', 'Analysis',  'Map', 'Prediction', 'Team'], 
  #      icons=['house-fill','bar-chart-fill','globe','x-diamond-fill','person-fill'], 
   #     menu_icon="cast", default_index=0,
    #    styles={
     #       "container": {"padding": "0!important", "background-color": "#fafafa"},
      #      "icon": {"color": "orange", "font-size": "25px"}, 
       #     "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        #    "nav-link-selected": {"background-color": "green"}
        #}
    #)

# Load the data
df1 = pd.read_csv('merged_data1.csv')
df = df1.copy()

if selected == 'Home':


    st.write(" ")
    st.header(':blue[Project background]')
    st.subheader(':blue[Introduction:]')
    st.write("""
                
                Global household electrification refers to the proportion of households worldwide with access to electricity,
                crucial for poverty reduction, economic growth, and improved living standards. While lacking a universally agreed definition,
                it generally includes reliable power, cooking facilities, and minimum consumption. The International Energy Agency's framework
                sets evolving benchmarks based on urban/rural settings. Progress has been made, with global electricity access rising from 71% 
                in 1990 to 87% in 2016, largely due to infrastructure and policy advancements. Despite OECD countries nearing universal access, 
                inequality remains, especially in underdeveloped areas where 13% lacked electricity in 2016. Bridging this gap is a shared 
                responsibility for governments, international bodies, and stakeholders in pursuit of sustainable development and inclusivity. 
                (Ritchie et. al, 2022) 

             
             """)
    
    st.header(':blue[Problem Statement ]')

    st.write("""
                Extensive research has been conducted on household electrification challenges in Sub-Saharan Africa, the variations in 
                access to electricity within OECD (Organization for Economic Cooperation and Development) countries remain an underexplored
                area of study. Despite being characterized by high levels of economic development, certain segments of populations within
                OECD countries continue to face inadequate access to electricity, hindering their socio-economic progress and well-being.
                The current electrification datasets present a significant challenge due to their inconsistency, inaccuracy, and limited 
                availability. These shortcomings have the potential to impede informed and effective decision-making processes.  

             
             """)
    

    st.header(':blue[Aim and Objectives]')
    st.write("""
                The primary aim of this project is to develop a global household predictive model with a specific focus solely on OECD countries.
                 While the objectives are:
                 to incorporate advanced machine learning techniques and additional dataset to refine the model’s accuracy.
                 to perform some analysis to uncover patterns, correlations, and trends within the dataset to extract some valuable information. 

            
            
            
            """)
    

    st.write(" ")
    
    st.markdown("For further details, check [Team GCP Capstone Research Paper](https://docs.google.com/document/d/18_uIrVqsfzAET3wjd7lRaoK8dZETpv-eZ2sBKlTRNak/edit)")

    
elif selected == 'Visualizations':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title(":blue[Data Visualizations]")

    st.subheader(":green[Histogram and Kernel Density Plot of Value]")
    # Reset the index
    df1.reset_index(drop=True, inplace=True)
    # Set the style of the plots
    sns.set(style="whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 2))

    # Histogram
    sns.histplot(df1['Value'], bins=20, kde=False, ax=axes[0])
    axes[0].set_title('Histogram of Value')

    # Kernel Density Plot
    sns.kdeplot(df1['Value'], ax=axes[1])
    axes[1].set_title('Kernel Density Plot of Value')

    # Adjust layout
    st.pyplot(fig) 
    
    st.subheader(":green[Distribution Plots]")
    # Distribution Plots
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    sns.histplot(df1['rural pop'], kde=True)
    plt.title("Distribution of Rural Population (%)")

    plt.subplot(1, 3, 2)
    sns.histplot(df1['urban pop'], kde=True)
    plt.title("Distribution of Urban Population (%)")

    plt.subplot(1, 3, 3)
    sns.histplot(df1['Value'], kde=True)
    plt.title("Distribution of Value")

    plt.tight_layout()
    st.pyplot() 

    st.subheader(":green[Line Plot of Total Value Over Time]")

    # Line Plot: Total Value Over Time
    # Group by Location and Year and calculate mean values
    grouped_df = df1.groupby(['Location', 'Year']).sum().reset_index()
    plt.figure(figsize=(10,2))
    sns.lineplot(data=grouped_df, x='Year', y='Value')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Line Plot of Total Value Over Time')
    st.pyplot() 

    st.markdown("For more visualizations, check [Team GCP Capstone Research Paper](https://docs.google.com/document/d/18_uIrVqsfzAET3wjd7lRaoK8dZETpv-eZ2sBKlTRNak/edit)")

    

elif selected == 'Prediction':
    model=pickle.load(open('model.pkl','rb'))

    # Initialize LabelEncoder for 'Location'
    label_encoder = LabelEncoder()
    df['Location'] = label_encoder.fit_transform(df['Location'])

    # Initialize StandardScaler
    scaler = StandardScaler()

    scaler.fit(df[['Location', 'Year', 'rural pop', 'urban pop', 'electric_rural']])

    # function to make predictions
    def make_prediction(location, year, rural_population, urban_population, electric_rural):
        # Create DataFrame
        input_data = pd.DataFrame({
            'Location': [location],
            'Year': [year],
            'rural pop': [rural_population],
            'urban pop': [urban_population],
            'electric_rural': [electric_rural]
        })
        # Load label mapping
        label_mapping = {0: "High", 1: "Low", 2: "Medium"} 
        # Encode 'Location' using the label encoder
        input_data['Location'] = label_encoder.transform(input_data['Location'])
        # Scale the user input data using the scaler
        user_input_scale = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
        
        prediction = model.predict(user_input_scale)
        
        predicted_class_label = label_mapping[prediction[0]]  # Map to unencoded label
        
        
        return predicted_class_label

    # Streamlit app
    st.markdown("## Electricity Class Prediction")
    st.markdown("### This a multiclassification prediction of 'Low', 'Medium' and 'High.'")
    st.markdown("#### Range of Low: 3.5GWh - 51,569.333GWh")
    st.markdown("#### Range of Medium: 51,569.333GWh - 180,212.259GWh")
    st.markdown("#### Range of High: 180,212.259GWh - 4,190,552.0GWh")

    
    # Input fields for user
    location = st.selectbox("Select Location", df1['Location'].unique())
    year = st.number_input("Select Year", value=2023, step=1)
    rural_population = st.number_input("Select Rural Population (%)", min_value=0.0000, max_value=100.0000)
    #urban_population = st.number_input("Select Urban Population (%)", min_value=0.0000, max_value=100.0000)

    urban_population = 100.0000 - rural_population
    st.info(f"Urban Population (%): {urban_population}")
    electric_rural = st.number_input("Select electric rural", min_value=0.0000)

    # Predict button
    if st.button("Predict"):
        predicted_class_label = make_prediction(location, year, rural_population, urban_population, electric_rural)
        print(predicted_class_label)
        # Map the class label to a more descriptive message
        if predicted_class_label == "Low":
            predicted_message = "Low level household electricity value"
        elif predicted_class_label == "Medium":
            predicted_message = "Medium level household electricity value"
        elif predicted_class_label == "High":
            predicted_message = "High level household electricity value"
        else:
            predicted_message = "Unknown"

        import time
        with st.spinner('Wait for a seconds...'):
            time.sleep(1)

        progress_bar = st.progress(0)
        # Simulate a task that takes time
        for i in range(101):
            time.sleep(0.001)  # Simulate some processing time
            progress_bar.progress(i)    
        st.write(f"{location} will have a {predicted_message} in the year {year}")

elif selected == "Team":

    st.write(" ")
    st.header(':blue[List of Active Members]')


    st.markdown("""
                    - Aramide Adebesin
                    - Kayode Jesujana
                    - Obinna Nwachukwu
                    - Ridwan Abdurahman
                    - David Udosen
                    - Godwin Ehizojie Michael
                    - Emmanuel Christopher
                    - Gopal Kumar
                    - Oluwatomisin Arokodare
                    - Abdul Basit Solkar
                    - Elizabeth Abodunrin
                    - Gloriah Mwongeli Munyao
                    - Adedolapo Ogunlade
                    - Nivedha S
                """)
    st.write("""
                
                
                
                



            """)
