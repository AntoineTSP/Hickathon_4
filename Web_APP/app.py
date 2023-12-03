import streamlit as st
from model import *
import pandas as pd
from PIL import Image
from codecarbon import track_emissions
from joblib import dump, load

@track_emissions()
def app():

    # Apply the custom theme
    st.set_page_config(
        page_title="Your ML App",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚀 Supply-AI")

    # Custom sidebar content
    st.sidebar.title("Personalized itinerary")
    selected_option = st.sidebar.radio("Your profile", ["Internal expertises", "Customers", "Investors"])


    # Main content based on selected option
    if selected_option == "Internal expertises":
        st.title("Welcome back colleague")
        st.write("Let's improve our supply-chain !")
        st.markdown("## Predictions from Uploaded CSV 📊")
        # File upload
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:

            # Read the CSV file
            df_emission = pd.read_csv("emissions.csv", sep=",")
            df_emission_limited = df_emission[["timestamp", "project_name","duration", "emissions",
            "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed", "country_name"]]
            df_power = df_emission[["duration", "cpu_power", "gpu_power", "ram_power"]]
            df_power["Nb_light_bulb"] = (df_power["cpu_power"] + df_power["gpu_power"] + df_power["ram_power"])/60
            df_energy = df_emission[["duration" , "cpu_energy", "gpu_energy" , "ram_energy", "energy_consumed"]]
            df_energy["cpu_energy"] = df_energy["cpu_energy"] *1000
            df_energy["gpu_energy"] = df_energy["gpu_energy"] *1000
            df_energy["ram_energy"] = df_energy["ram_energy"] *1000
            df_energy["energy_consumed"] = df_energy["energy_consumed"] *1000
            df_energy["Duration_light_bulb_milisec"] = df_energy["energy_consumed"] *1000 /(1*24) 

            col1, col2 = st.columns(2)
            # Format the number using Markdown with custom CSS styling
            nb= round(df_energy["Duration_light_bulb_milisec"].iloc[-2],2)
            col1.markdown("<div style='font-size:1em;color:#424242;width:100%;text-align:center;'>Delivering your supply-chain state is equivalent of lighting up a bulb for: </div>", unsafe_allow_html=True)
            fancy_number = "<div style='font-size:7em;color:#FFD700;width:100%;text-align:center;'>" + str(nb) + "s</div>"
            col1.markdown(fancy_number, unsafe_allow_html=True)
            col2.image("bulb.jpg", width=300, use_column_width=False)

            #st.markdown("### Total sales per country:") 
            #st.image('./Total_sales_country.png')
            st.markdown("### Total sales per country with CO2 emissions:") 
            st.image('./Total_sales_co2.png')
            st.markdown("### Total stocks per country:") 
            st.image('./Total_Stocks.png')



            # Read the CSV file
            #df = pd.read_csv(uploaded_file, sep=",")
            df = pd.read_csv("~/hfactory_magic_folders/groups/11_none_shared_workspace/datasets/X_test_working.csv")
            
            df = pd.read_csv("~/hfactory_magic_folders/groups/11_none_shared_workspace/datasets/X_test_working.csv")
            gspci = pd.read_csv("~/hfactory_magic_folders/groups/11_none_shared_workspace/datasets/extra-dataset/GSCPI_data.csv", sep=",")
            gspci['Year'] = gspci['Year-Month'].apply(lambda x: x.split('-')[0]).astype(int)
            gspci['Month'] = gspci['Year-Month'].apply(lambda x: x.split('-')[1]).astype(int)
            gspci.drop(columns=['Year-Month'], inplace=True)

            clf = load('RFR_0.51.joblib') 
            enc = load('target_enc_RFR_051.joblib')

            real_test = pd.read_csv("~/hfactory_magic_folders/groups/11_none_shared_workspace/datasets/X_test_working.csv")
            real_test.fillna(0, inplace=True)
            add_gscpi_to_df(real_test, gspci)

            for j in range(1, 4):
                real_test[f'Month {j}'] = real_test[f'Month {j}'].apply(lambda x: to_int(x))
            
            features_to_keep = ['Site', 'Reference proxy', 'Customer Persona proxy', 'Strategic Product Family proxy', 'Date', 'Month 1', 'Month 2', 'Month 3', 'gscpi']
            real_test = real_test[features_to_keep]
            index_fst_not_encoded = list(real_test.columns).index('Month 1')
            real_test_not_encoded = real_test[real_test.columns[index_fst_not_encoded:]]
            real_test = enc.transform(real_test[real_test.columns[:index_fst_not_encoded]])
            real_test = np.hstack((real_test, real_test_not_encoded))
            real_pred = clf.predict(real_test)
            real_test_results = pd.DataFrame()
            real_test_results['index'] = pd.read_csv("~/hfactory_magic_folders/groups/11_none_shared_workspace/datasets/X_test_working.csv")['index'].values
            real_test_results['Month 4'] = real_pred

            # Display the uploaded data
            #st.write("Uploaded DataFrame:")
            st.markdown("### Uploaded DataFrame:")
            st.write(df[:100].style.text_gradient(axis = 0,cmap="bwr"))

            # Perform predictions on the uploaded data
            predictions = real_test_results
            st.markdown("### What is relevant to predict future sales:") 
            st.image('./feature_selecion.png', caption="Our features selection", width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            st.markdown("### Predicted DataFrame:") 
            st.write(predictions[1000:1100].style.text_gradient(axis = 0,cmap="bwr"))

            get_bar_chart(predictions[1000:1100])
            #get_time_series(df)

            # Button to download the DataFrame as CSV
            if st.button("Download the data as CSV"):
                csv_string = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_string,
                    file_name="downloaded_data.csv",
                    key="download_dataframe",
                    help="Click to download the current DataFrame as CSV.",
                )

            # Button to download the plot
            if st.button("Download Plot"):
                # Save the plot as an image file
                #fig.savefig("downloaded_plot.png", format="png")
                #im = Image.open("./standard_plot.png")

                # Offer the saved plot for download
                st.download_button(
                    label="Download Plot",
                    data="standard_plot.png",
                    file_name="downloaded_plot.png",
                    key="download_plot",
                    help="Click to download the current plot as an image.",
                )
            #st.write("Emissions all:")
            #st.write(df_emission.style.text_gradient(axis = 0,cmap="bwr"))
            st.write("Emissions limited:")
            st.write(df_emission_limited.style.text_gradient(axis = 0,cmap="bwr"))
            st.write("Power :")
            st.write(df_power.style.format('{:.4f}', na_rep="")\
                .bar(align=0, cmap='OrRd', height=50,
                    width=60, props="width: 120px; border-right: 1px solid black;")\
                .text_gradient(axis = 0,cmap="bwr"))
            st.write("Energy :")
            st.dataframe(df_energy.style.format('{:.4f}', na_rep="")\
                .bar(align=0, cmap='OrRd', height=50,
                    width=60, props="width: 120px; border-right: 1px solid black;")\
                .text_gradient(axis = 0,cmap="bwr"))
            
            

    elif selected_option == "Customers":
        st.title("You trust our reliable suppy-chain : see why you made the good choice")
        st.markdown("## State of your command 📊")
        st.write("Custom content for Option 2 goes here.")
    elif selected_option == "Investors":
        st.title("You trust our reliable suppy-chain : see why we can reach the sky together")
        st.markdown("## State of our supply-chain📊")
        st.write("Custom content for Option 3 goes here.")
app()