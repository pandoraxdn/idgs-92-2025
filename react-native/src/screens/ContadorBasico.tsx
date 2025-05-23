import React from 'react';
import { View, Text, Button } from 'react-native';
import { useContadorBasicoHook } from '../hooks/useContadorBasicoHook';
import { appTheme } from '../themes/appTheme';

export const ContadorBasico = () => {

    const { contador, add, dec, reset } = useContadorBasicoHook();

    return(
        <View
            style={ appTheme.container }
        >
            <Text
                style={ appTheme.text }
            >
                Contador: { contador }
            </Text>
            <Button
                title='Add+'
                onPress={ () => add() }
            />
            <Button
                title='Decrement+'
                onPress={ () => dec() }
            />
            <Button
                title='Reset'
                onPress={ () => reset() }
            />

        </View>
    )
}
