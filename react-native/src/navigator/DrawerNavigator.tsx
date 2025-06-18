import { useContext } from 'react';
import { createDrawerNavigator } from "@react-navigation/drawer";
import { UserNavigator } from "./UserNavigator";
import { useWindowDimensions } from "react-native";
import { DrawerMenu } from "../components/DraweMenu";
import { ContadorBasico } from './../screens/ContadorBasico';
import { PanecitoScreen } from './../screens/PanecitoScreen';
import { AuthContext } from './../context/AuthContext';
import { LoginScreen2 } from '../screens/users/LoginScreen2';
import { HomeScreen } from '../screens/HomeScreen';

export type RootDrawerProps = {
    UserNavigator: undefined;
    ContadorBasico: undefined;
    PanecitoScreen: undefined;
    HomeScreen:     undefined;
}

const Navigator = () => {

    const Drawer = createDrawerNavigator<RootDrawerProps>();

    const { width } = useWindowDimensions();

    return (
        <Drawer.Navigator
            initialRouteName="UserNavigator"
            screenOptions={{
                headerShown: true,
                drawerType: (width >= 768) ? 'permanent' : 'front',
                overlayColor: "rgba(242,87,35,0.4)",
                drawerPosition: "left",
                drawerStyle: {
                    backgroundColor: 'rgba(200,200,200,0.7)',
                    width: width *0.6
                }
            }}
            drawerContent={ (props) => <DrawerMenu {...props}/> }
        >
            <Drawer.Screen
                name="UserNavigator"
                component={UserNavigator}
            />
            <Drawer.Screen
                name="PanecitoScreen"
                component={PanecitoScreen}
            />
            <Drawer.Screen
                name="ContadorBasico"
                component={ContadorBasico}
            />
            <Drawer.Screen
                name="HomeScreen"
                component={HomeScreen}
            />
        </Drawer.Navigator>
    );
    
}

export const DrawerNavigator = () => {

    const { authState } = useContext(AuthContext);

    return authState.isLoggedIn ? <Navigator/> : <LoginScreen2/>

}
