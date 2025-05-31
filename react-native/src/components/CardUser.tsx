import React from 'react';
import { useNavigation } from "@react-navigation/native";
import { View, Text, StyleSheet, Dimensions, TouchableOpacity } from 'react-native';
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
                    backgroundColor: type_user(user),
                    width: width * 0.4
                }}
            >
                <Text
                    style={ style.cardTitle }
                >
                    { `Título:\n ${ user.username }` }
                </Text>
            </View>
        </TouchableOpacity>
    )
}

const style = StyleSheet.create({
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

