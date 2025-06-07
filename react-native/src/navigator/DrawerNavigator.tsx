import { createDrawerNavigator } from "@react-navigation/drawer";
import { UserNavigator } from "./UserNavigator";
import { useWindowDimensions } from "react-native";
import { DrawerMenu } from "../components/DraweMenu";

export type RootDrawerProps = {
    UserNavigator: undefined;
}

const Navigator = () => {

    const Drawer = createDrawerNavigator<RootDrawerProps>();

    const { width } = useWindowDimensions();

    return (
        <Drawer.Navigator
            screenOptions={{
                headerShown: true,
                drawerType: (width >= 768) ? 'permanent' : 'front',
                overlayColor: "rgba(242,87,35,0.4)",
                drawerPosition: "right",
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
        </Drawer.Navigator>
    );
    
}

export const DrawerNavigator = () => {
    return(
        <Navigator/>
    );
} 
