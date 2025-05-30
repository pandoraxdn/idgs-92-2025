import React, { useReducer } from 'react';
import { View, Text, Button } from 'react-native';
import { appTheme } from '../themes/appTheme';

interface AuthState{
    counter: number;
}

export const UseReducerScreen = () => {

    const initialSate: AuthState = {
        counter: 10,
    }

    type Action = 
        | { type: "add" } 
        | { type: "dec" } 
        | { type: "reset" };

    const counterReducer = ( state: AuthState, action: Action ) => {
        switch( action.type ){
            case 'add':
                return { counter: state.counter + 1 };
            case 'dec':
                return { counter: (state.counter == 0 ) ? 0 : state.counter -1 };
            case 'reset':
                return { counter: initialSate.counter };
        }
    }

    const [ estado, dispatch ] = useReducer(counterReducer,initialSate);


    return(
        <View
            style={ appTheme.container }
        >
            <Text
                style={ appTheme.text }
            >
                Contador: { estado.counter }
            </Text>
            <Button
                title='Add+'
                onPress={ () => dispatch({ type: "add" }) }
            />
            <Button
                title='Decrement+'
                onPress={ () => dispatch({ type: "dec" }) }
            />
            <Button
                title='Reset'
                onPress={ () => dispatch({ type: "reset" }) }
            />
        </View>
    )
}
