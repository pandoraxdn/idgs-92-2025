import React, { useEffect, useState } from 'react';
import { View, Text } from 'react-native';
import { appTheme } from '../themes/appTheme';

export const PanecitoScreen = () => {

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
                    ...appTheme.text,
                    color: bgColor
                }}
            >
                { hora.toLocaleString() }
            </Text>
        </View>
    )
}
