import { StyleSheet } from "react-native";

export const appTheme = StyleSheet.create({
    container:{
        justifyContent: "center",
        alignItems: "center",
        alignContent: "center",
        flex: 1
    },
    text:{
        fontSize: 60,
        fontWeight: "bold",
        textAlign: "center",
        color: "violet"
    },
    subtext:{
        color: "black",
        fontSize: 30,
        textAlign: "left",
        marginTop: 5,
        marginLeft: 12
    },
    input:{ 
        backgroundColor: "violet",
        borderWidth: 5,
        borderColor: "pink",
        borderRadius: 10,
        color: "white",
        textAlign: "center",
        fontWeight: "bold",
        height: 70,
        width: 280,
        margin: 12,
        fontSize: 30
    },
    btn:{
        alignSelf: "center",
    }
});
