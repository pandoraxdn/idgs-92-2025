import { AuthState } from './AuthContext';

type AuthActions = 
    | { type: "singIng" }
    | { type: "logout" }
    | { type: "changeUserName", payload: string }
    | { type: "changeAvatar", payload: string };

export const authReducer = ( state: AuthState, action: AuthActions ) => {
    switch(action.type){
        case "singIng":
            return {
                ...state,
                isLoggenIn: true,
                username: "pandora",
                avatar: "image.jpg"
        };
        case "logout":
            return {
                ...state,
                isLoggenIn: false,
                username: "",
                avatar: ""
        };
        case 'changeUserName':
            return{
                ...state,
                username: action.payload
        };
        case 'changeAvatar':
            return{
                ...state,
                avatar: action.payload
        };
    }
}
