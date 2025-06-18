import React, { useEffect, useState, useContext } from 'react';
import { View, Text } from 'react-native';
import { appTheme } from '../themes/appTheme';
import { AuthContext } from '../context/AuthContext';

export const PanecitoScreen = () => {

    const { authState } = useContext(AuthContext);

    const [ hora, setHora ] = useState( new Date() );

    const [ bgColor, setBgColor ] = useState<string>();

    const colors:string[] = [ "violet", "purple", "orange", "black" ];

    const random = () => {
        const color = colors[ Math.floor(Math.random() * colors.length) ];
        setBgColor(color);
    }

    useEffect( () => {
        const interval = setInterval(() => {
            setHora( new Date() );
            return () => clearInterval(interval);
        },1000);

        const intervalColor = setInterval(() => {
            random();
            return () => clearInterval(intervalColor);
        },100);
    },[]);

    return(
        <View
            style={ appTheme.container }
        >
            <Text
                style={{
                    fontSize: 30
                }}
            >
                { JSON.stringify(authState) }
            </Text>
            <Text
                style={{
                    ...appTheme.text,
                    color: bgColor
                }}
            >
                { hora.toLocaleString() }
            </Text>
        </View>
    )
}
