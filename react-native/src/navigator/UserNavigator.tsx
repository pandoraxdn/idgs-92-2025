import React from "react";
import { createStackNavigator } from "@react-navigation/stack";
import { HomeUserScreen } from "../screens/users/HomeUserScreen";
import { FormScreen } from "../screens/users/FormUserScreen";
import { UsersData } from "../interfaces/requestApi";

export type RootStackUserParams = {
    HomeUserScreen: undefined;
    FormScreen: { user: UsersData };
}

export const UserNavigator = () => {

    const Stack = createStackNavigator<RootStackUserParams>();

    return(
        <Stack.Navigator
            initialRouteName="HomeUserScreen"
            screenOptions={{
                headerMode: 'float',
                headerShown: true,
                headerStyle: {
                    height: 100,
                    shadowColor: "violet",
                    backgroundColor: "pink",
                    borderWidth: 5,
                    borderColor: "gray",
                    borderRadius: 20,
                    opacity: 0.8
                },
                headerTitleStyle:{
                    fontWeight: "bold",
                    color: "white"
                },
                headerTintColor: "white",
                cardStyle: {
                    backgroundColor: "white"
                }
            }}
        >
            <Stack.Screen
                name="HomeUserScreen"
                component={HomeUserScreen}
                options={{ title: "Index de Usuarios" }}
            />
            <Stack.Screen
                name="FormScreen"
                component={FormScreen}
                options={{ title: "Formulario de Usuarios" }}
            />
        </Stack.Navigator>
    );
}


