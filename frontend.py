import streamlit as st


import streamlit as st

# display options when the user selects maps
def display_options():
    st.write("Select an option from below to get started:")
    option = st.radio(
    "Which match do you want to predict?",
    ('', Team1(), Team2()),
    index = 0
    )
    if option == '':
        st.write("You cannot select an empty string")
    else :
        st.write("Option: ",option)
    return option


def Team1():
    return "TEAM A  vs TEAM B"

def Team2():
    return "TEAM C vs  TEAM D"

if __name__ == "__main__":
    st.sidebar.markdown(
        "<h1 style='color: red; font-size: 36px; margin-top: 30px;'>About ValoStats</h1>", 
        unsafe_allow_html=True
    )

    st.sidebar.write("""
    ValoStats is your go-to tool for mastering Valorant. 
    """)


    st.markdown(
        "<h1 style='font-size: 36px; margin-top: 50px;'>Welcome to ValoStats!</h1>", 
        unsafe_allow_html=True
    )

    options = display_options()
    if (options):
        for i in range(11):
            st.write(i)



