import { connect, StringCodec, type NatsConnection } from "nats";

const DEFAULT_NATS_URL = "nats://127.0.0.1:4222";
const stringCodec = StringCodec();

let connectionPromise: Promise<NatsConnection> | null = null;

async function createConnection() {
	const servers = process.env.NATS_URL ?? DEFAULT_NATS_URL;
	const connection = await connect({ servers });

	connection.closed().then((err) => {
		if (err) {
			console.error("NATS connection closed due to error", err);
		}
		connectionPromise = null;
	});

	if (typeof process !== "undefined") {
		const cleanup = async () => {
			if (connectionPromise) {
				await connection.close();
			}
		};

		process.once("beforeExit", cleanup);
		process.once("SIGINT", cleanup);
		process.once("SIGTERM", cleanup);
	}

	return connection;
}

export async function getNatsConnection() {
	if (!connectionPromise) {
		connectionPromise = createConnection();
	}
	return connectionPromise;
}

export async function publishJson(subject: string, payload: unknown) {
	const connection = await getNatsConnection();
	const encoded = stringCodec.encode(JSON.stringify(payload));
	await connection.publish(subject, encoded);
}
