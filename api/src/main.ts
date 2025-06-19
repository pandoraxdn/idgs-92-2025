import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { json, urlencoded } from 'body-parser';
import { networkInterfaces } from 'os';

const getLocalIp = () =>
  Object.values(networkInterfaces())
    .flat()
    .find(i => i?.family === 'IPv4' && !i.internal)?.address || 'localhost';

async function main() {
  const app = await NestFactory.create(AppModule);
  app.setGlobalPrefix("api/v1");
  app.use(json({ limit: "100mb" }));
  app.use(urlencoded({ limit: "100mb", extended: true }));
  const PORT = process.env.PORT || 3000;
  await app.listen(PORT);
  console.log(`API: http://${getLocalIp()}:${PORT}`);
}
main();
