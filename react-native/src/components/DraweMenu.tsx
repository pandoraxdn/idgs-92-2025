import React from 'react';
import { DrawerContentScrollView, DrawerContentComponentProps } from '@react-navigation/drawer';
import { View, Image, Text } from 'react-native';
import { BtnTouch } from './../components/BtnTouch';
import { appTheme } from '../themes/appTheme';

export const DrawerMenu = ( { navigation }: DrawerContentComponentProps ) => {

    return(
        <DrawerContentScrollView>
            <View
                style={{
                    justifyContent: "center",
                    alignItems: "center"
                }}
            >
                <Image
                    style={{
                        height: 200,
                        width: 200,
                        borderColor: "white",
                        borderWidth: 5,
                        borderRadius: 100
                    }}
                    source={{
                        uri: 'https://cdn.pixabay.com/photo/2023/03/25/23/58/capybara-7877166_640.png'
                    }}
                />
                <Text
                    style={{
                        ...appTheme.text,
                        color: "black",
                        fontSize: 20
                    }}
                >
                    Username: "Juanito"
                </Text>
                <BtnTouch
                    background='violet'
                    onPress={ () => navigation.navigate("UserNavigator") }
                    bColor='pink'
                    title='Crud Usuarios'
                />
                <BtnTouch
                    background='violet'
                    onPress={ () => navigation.navigate("PanecitoScreen") }
                    bColor='pink'
                    title='Panecito'
                />
                <BtnTouch
                    background='violet'
                    onPress={ () => navigation.navigate("ContadorBasico") }
                    bColor='pink'
                    title='ContadorBasico'
                />
                <BtnTouch
                    background='violet'
                    onPress={ () => navigation.navigate("HomeScreen") }
                    bColor='pink'
                    title='HomeScreen'
                />
            </View>
        </DrawerContentScrollView>
    );
}


