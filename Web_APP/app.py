import streamlit as st
from model import *
import pandas as pd
from PIL import Image
from codecarbon import track_emissions

@track_emissions()
def app():
    X,y,X_test = generate_data()
    # convert array into dataframe
    DF = pd.DataFrame(X_test)
    # save the dataframe as a csv file
    DF.to_csv("X_test.csv",index=False)

    model= fit(X, y)
    y_hat=predict(model, X_test)
    plot(X, y, X_test, y_hat)


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
            df = pd.read_csv(uploaded_file, sep=",")

            # Display the uploaded data
            st.write("Uploaded DataFrame:")
            st.write(df)

            # Perform predictions on the uploaded data
            predictions = model.predict(df)  # Assuming your predict function accepts a DataFrame
            st.write("Predictions:")
            st.write(predictions)

            get_chart(df)
            get_time_series(df)

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

    elif selected_option == "Customers":
        st.title("You trust our reliable suppy-chain : see why you made the good choice")
        st.markdown("## State of your command 📊")
        st.write("Custom content for Option 2 goes here.")
    elif selected_option == "Investors":
        st.title("You trust our reliable suppy-chain : see why we can reach the sky together")
        st.markdown("## State of our supply-chain📊")
        st.write("Custom content for Option 3 goes here.")

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
    df_energy["Duration_light_bulb_milisec"] = df_energy["energy_consumed"] *1000 /(1000*24) 

    col1, col2 = st.columns(2)
    # Format the number using Markdown with custom CSS styling
    nb= round(df_energy["Duration_light_bulb_milisec"].iloc[-1],4)
    col1.markdown("<div style='font-size:1em;color:#424242;width:100%;text-align:center;'>Delivering your supply-chain state is equivalent of lighting up a bulb for: </div>", unsafe_allow_html=True)
    fancy_number = "<div style='font-size:7em;color:#FFD700;width:100%;text-align:center;'>" + str(nb) + "ms</div>"
    col1.markdown(fancy_number, unsafe_allow_html=True)
    col2.image("bulb.jpg", width=300, use_column_width=False)

    st.write("Emissions all:")
    st.write(df_emission.style.text_gradient(axis = 0,cmap="bwr"))
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
app()