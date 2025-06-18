import React, { useContext } from 'react';
import { View, Text } from 'react-native';
import { appTheme, theme } from '../themes/appTheme';
import { AuthContext } from '../context/AuthContext';
import { BtnTouch } from '../components/BtnTouch';

export const HomeScreen = () => {

    const { authState, changeTheme } = useContext(AuthContext);

    return(
        <View
            style={{
                ...appTheme,
                flex: 1,
                backgroundColor: ( authState.theme == "dark" ) ? theme.darkTheme.backgroundTheme : theme.ligthTheme.backgroundTheme
            }}
        >
            {
                (authState.theme == "dark")
                ?
                    <BtnTouch
                        title='Obscuro'
                        background={ (authState.theme == "dark") ? theme.darkTheme.colorBtn : theme.ligthTheme.colorBtn }
                        onPress={() => changeTheme("dark") }
                        bColor='white'
                    />
                :
                    <BtnTouch
                        title='Claro'
                        background={ (authState.theme == "dark") ? theme.darkTheme.colorBtn : theme.ligthTheme.colorBtn }
                        onPress={() => changeTheme("light") }
                        bColor='black'
                    />
            }
            <Text
                style={{
                    color: (authState.theme == "dark") ? theme.darkTheme.titleC : theme.ligthTheme.titleC
                }}
            >
                { JSON.stringify(authState) }
            </Text>
        </View>
    )
}
