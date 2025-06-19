import { useReducer, useState, useContext } from "react";
import { pandoraApi } from "../api/pandoraApi";
import { RequestLogin } from "../interfaces/requestApi";
import { AuthContext } from "../context/AuthContext";

export interface LoginData{
    username: string;
    password: string;
}

const initialState: LoginData = {
    username: "",
    password: ""
}

type Action = { type: "handleInputChange", payload: { fieldName: keyof LoginData, value: string } };

const dataReducer = (state: LoginData, action: Action) => {
    switch( action.type){
        case 'handleInputChange':{
            return {
                ...state,
                [ action.payload.fieldName ] : action.payload.value
            }
        }
    }
}

export const useLoginUser = () => {

    const [ loading, setLoading ] = useState<boolean>(false);
    const [ state, dispatch ] = useReducer( dataReducer, initialState );
    const [ request, setRequest ] = useState<RequestLogin>();
    const { singIng, changeAvatar, changeUserName } = useContext(AuthContext);

    const handleInputChange = ( fieldName: keyof LoginData, value: string ) => {
        dispatch({ type: "handleInputChange", payload: {fieldName, value} });
    }

    const handleLogin = async () => {
        setLoading(true);
        const apiUrl = 'http://192.168.1.74:3000/api/v1/users/login';
        const dataBody ={
            username: state.username,
            password: state.password
        }
        try{
            const response = await pandoraApi.post(apiUrl,dataBody);
            (response.data !== false) && ( () => {
                singIng();
                changeUserName(response.data.username);
                changeAvatar(response.data.imagen);
                setRequest(response.data);
            })();
        }catch(error){
            console.log( "Error", error );
            setRequest(false);
        }
        setLoading(false);
    }

    return { loading, state, handleLogin, handleInputChange, request };
}


