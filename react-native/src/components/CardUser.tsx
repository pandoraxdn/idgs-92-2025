import React from 'react';
import { useNavigation } from "@react-navigation/native";
import { View, Text, StyleSheet, Dimensions, TouchableOpacity, Image } from 'react-native';
import { UsersData } from '../interfaces/requestApi';

interface Props{
    user: UsersData;
}

export const CardUser = ( { user }: Props ) => {

    const navigation = useNavigation();

    const { width } = Dimensions.get("window");

    const type_user = ( user: UsersData ) => {
        switch( user.tipo ){
            case "user":
                return "#e8c70b";
            case "client":
                return "#90EE90";
            case "admin":
                return "#d90000";
        }
    }

    return(
        <TouchableOpacity
            key={ `${user._id}${user.__v}`}
            activeOpacity={0.9}
            onPress={ () => navigation.navigate("FormScreen",{user:user}) }
        >
            <View
                style={{
                    ...style.cardContainer,
                    width: width * 0.4,
                    overflow: "hidden"
                }}
            >
                <View
                    style={{
                        ...style.backgroundTop,
                        backgroundColor: type_user(user),
                    }}
                />
                <View
                    style={{
                        ...style.backgroundBottom
                    }}
                />

                <Text
                    style={ style.cardTitle }
                >
                    { `Username:\n ${ user.username }\n` }
                    { `Tipo:\n ${ user.tipo }` }
                </Text>
                <Image 
                    source={{
                        uri: `data:image/jpeg;base64,${user.imagen}`
                    }}
                    style={{
                        right: -10,
                        bottom: -10,
                        borderColor: "white",
                        borderWidth: 3,
                        width: 80,
                        height: 80,
                        borderRadius: 20,
                        position: "absolute",
                    }}
                />
            </View>
        </TouchableOpacity>
    )
}

const style = StyleSheet.create({
    backgroundTop:{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,        
        bottom: "50%",
        backgroundColor: "blue",
        transform:[
            { rotateX: "20deg" },
            { rotateY: "-45deg" },
            { scale: 2 }
        ]
    },
    backgroundBottom:{
        position: "absolute",
        top: "50%",
        left: 0,
        right: 0,        
        bottom: 0,
        backgroundColor: "gray",
        transform:[
            { rotateX: "20deg" },
            { rotateY: "-45deg" },
            { scale: 2 }
        ]
    },
    cardContainer:{
        marginHorizontal: 10,
        height: 120,
        width: 120,
        marginBottom: 25,
        borderRadius: 20,
    },
    cardTitle:{
        marginTop: 10,
        marginHorizontal: 15,
        color: "white",
        fontSize: 25,
        fontWeight: "bold"
    }
});

