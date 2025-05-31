import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

interface Props{
    onPress:        () => void;
    background?:    string;
    title:          string;
    bColor:         string;
}

export const BtnTouch = ( { onPress, background='pink', title, bColor='gray' }:Props ) => {

    return(
        <TouchableOpacity
            onPress={ onPress }
        >
            <View
                style={{
                    ...style.btnCointaner,
                    backgroundColor: background,
                    borderColor: bColor
                }}
            >
                <Text
                    style={ style.btnTitle }
                >
                    { title }
                </Text>
            </View>
        </TouchableOpacity>
    );
}

const style = StyleSheet.create({
    btnCointaner:{
        borderRadius: 20,
        borderWidth: 6,
        marginBottom: 20,
        marginTop: 20,
        marginLeft: 20,
        marginRight: 20,
        justifyContent: "center",
        height: 50,
        width: 160
    },
    btnTitle:{
        color: "white",
        fontSize: 20,
        fontWeight: "bold",
        textAlign: "center"
    }
});

