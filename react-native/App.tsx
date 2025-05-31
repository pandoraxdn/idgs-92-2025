import React from 'react';
//import { Screen1 } from './src/screens/Screen1';
//import { ContadorBasico } from './src/screens/ContadorBasico';
//import { PanecitoScreen } from './src/screens/PanecitoScreen';
//import { UseReducerScreen } from './src/screens/UseReducerScreen';
//import { FormScreen } from './src/screens/FormScreen';
//import { ImagePickerScreen } from './src/screens/ImagePickerScreen';
import { NavigationContainer } from '@react-navigation/native';
import { UserNavigator } from './src/navigator/UserNavigator';

const App = () => {
    return (
        <NavigationContainer>
            <UserNavigator/>
        </NavigationContainer>
    );
}

export default App;
