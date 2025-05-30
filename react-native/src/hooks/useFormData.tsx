import { useState } from "react";

export interface FormData{
    username:   string;
    password:   string;
    imagen:      string;
    tipo:       "user" | "admin" | "client";
}

interface UseFormHook{
    formData: FormData;
    formList: FormData[];
    handleInputChange: ( fieldName: keyof FormData, value: string | number ) => void;
    handleSubmit: () => void;
}

export const useFormHook = (): UseFormHook => {
    const initialState: FormData = {
        username:   '',
        password:   '',
        imagen:     '',
        tipo:       'user',
    }

    const [ formData, setFormData ] = useState<FormData>(initialState);
    const [ formList, setFormList ] = useState<FormData[]>([]);
    
    const handleInputChange = ( fieldName: keyof FormData, value: string | number ) => {
        setFormData( (prevData) => ({
            ...prevData,
            [fieldName]: value
        }));
    }

    const handleSubmit = () => {
        setFormList( (prevList) => [...prevList, formData] );
        setFormData( initialState );
    }

    return {
        formData,
        formList,
        handleInputChange,
        handleSubmit
    }
}

