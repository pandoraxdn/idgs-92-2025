import React, { ReactNode } from 'react';
//import { Screen1 } from './src/screens/Screen1';
//import { ContadorBasico } from './src/screens/ContadorBasico';
//import { PanecitoScreen } from './src/screens/PanecitoScreen';
//import { UseReducerScreen } from './src/screens/UseReducerScreen';
//import { FormScreen } from './src/screens/FormScreen';
//import { ImagePickerScreen } from './src/screens/ImagePickerScreen';
import { NavigationContainer } from '@react-navigation/native';
import { DrawerNavigator } from './src/navigator/DrawerNavigator';
import { AuthProvider } from './src/context/AuthContext';

const App = () => {
    return (
        <NavigationContainer>
            <AppState>
                <DrawerNavigator/>
            </AppState>
        </NavigationContainer>
    );
}

const AppState = ( { children }: { children: ReactNode } ) => {
    return(
        <AuthProvider>
            { children }
        </AuthProvider>
    );

}

export default App;
