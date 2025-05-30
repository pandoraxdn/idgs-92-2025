import { IsNotEmpty, IsString, IsEnum, IsOptional } from "class-validator";
import { TypeUser } from "../schemas/user.schemas";

export class CreateUserDto {
    @IsString()
    @IsNotEmpty()
    username:   string;

    @IsString()
    @IsNotEmpty()
    password:   string;

    @IsString()
    @IsNotEmpty()
    imagen:       number;

    @IsEnum(TypeUser)
    @IsOptional()
    tipo:       TypeUser;
}
