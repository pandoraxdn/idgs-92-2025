import React from 'react';
import { View, Text } from 'react-native';
import { appTheme } from '../themes/appTheme';

export const Screen1 = () => {

    const cadena: string | number = "Daniela";

    const numero: number = 100

    const condicion: boolean = true;

    const arr: [number,string,boolean] = [1,"daniel",true];

    return (
        <View
            style={ appTheme.container }
        >
            <Text
                style={ appTheme.text }
            >
                Hola: {cadena}
            </Text>
            <Text
                style={ appTheme.text }
            >
                {numero}
            </Text>
            <Text
                style={ appTheme.text }
            >
                {( condicion ) ? "Verdadero" : "Falso"}
                { arr }
            </Text>
        </View>
    );
}
