import React, { useContext } from 'react';
import { View, Text, Image } from 'react-native';
import { AuthContext } from '../context/AuthContext';
import { appTheme, theme } from '../themes/appTheme';
import { BtnTouch } from '../components/BtnTouch';

export const HomeScreen = () => {

    const { authState, changeTheme } = useContext(AuthContext);

    return(
        <View
            style={{
                ...appTheme.container,
                flex: 1,
                backgroundColor: ( authState.theme == "light" ) ? theme.ligthTheme.backgroundTheme : theme.darkTheme.backgroundTheme,
            }}
        >

            <Image 
                source={{
                    uri: `data:image/png;base64,${authState.avatar}`
                }}
                style={{
                    width: 200,
                    height: 200,
                    borderRadius: 100,
                    borderWidth: 10,
                    borderColor: ( authState.theme == "light" ) ? theme.ligthTheme.titleC : theme.darkTheme.titleC,
                }}
            />
            <Text
                style={{
                    ...appTheme.text,
                    color: ( authState.theme == "light" ) ? theme.ligthTheme.titleC : theme.darkTheme.titleC,
                }}
            >
                User: { authState.username }
            </Text>
            <Text
                style={{
                    ...appTheme.text,
                    color: ( authState.theme == "light" ) ? theme.ligthTheme.titleC : theme.darkTheme.titleC,
                }}
            >
                Theme: { authState.theme }
            </Text>
            {
                (authState.theme == "light")
                ? <BtnTouch
                    title='Dark'
                    bColor={ theme.darkTheme.backgroundTheme }
                    onPress={ () => changeTheme("dark") }
                />
                : <BtnTouch
                    title='Light'
                    bColor={ theme.ligthTheme.backgroundTheme }
                    onPress={ () => changeTheme("light") }
                />
            }
        </View>
    );
}
