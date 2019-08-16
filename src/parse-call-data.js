'use strict'
const _ = require('lodash');
const chrono = require('chrono-node');
const crypto = require('crypto');
const fs = require('mz/fs');
const path = require('path');
const process = require('process');
const readline = require('readline');
const { promisify } = require('util');
const AbiEncoder = require('web3-eth-abi');
const yargs = require('yargs');

const ARGS = yargs
    .option('pretty', {
        type: 'boolean',
        describe: 'pretty print output',
    })
    .option('output', {
        alias: 'o',
        type: 'string',
        describe: 'output file',
    })
    .option('since', {
        alias: 's',
        type: 'string',
        describe: 'starting period',
    })
    .option('until', {
        alias: 's',
        type: 'string',
        describe: 'ending period',
    })
    .argv;

const glob = promisify(require('glob'));

const CALL_DATA_FILE = ARGS._[0];
const SINCE = chrono.parseDate(ARGS.since || 'july 30 2015').getTime() / 1000;
const UNTIL = chrono.parseDate(ARGS.until || new Date().toString()).getTime() / 1000;

(async () => {
    const abiFiles = await glob(path.join(__dirname, '../abis/**.json'));
    const callDecoder = new CallDecoder(await loadMergedAbiFiles(abiFiles));
    const rl = readline.createInterface({ input: fs.createReadStream(CALL_DATA_FILE)});
    const txs = {};
    const callers = {};
    let callCount = 0, orderCount = 0;
    rl.on('line', line => {
        const rawCall = JSON.parse(line);
        if (rawCall.timestamp < SINCE || rawCall > UNTIL) {
            return;
        }
        const { fromAddress } = rawCall;
        // If it's a direct call, the caller is the EOA, otherwise we pick the
        // the contract being called. This way we can group all calls that stem
        // from the same top-level contract, not just the immediate caller.
        const callerAddress = rawCall.toAddress == rawCall.calleeAddress ?
            rawCall.fromAddress : rawCall.toAddress;
        const info = callers[callerAddress] = callers[callerAddress] || {};
        const senders = info.senders = info.senders || {};
        const methods = info.methods = info.methods || {};
        const feeRecipients = info.feeRecipients = info.feeRecipients || {};
        const makers = info.makers = info.makers || {};
        info.caller = callerAddress;
        info.orderCount = info.orderCount || 0;
        info.fillCount = info.fillCount || 0;
        info.updateCount = info.updateCount || 0;
        senders[rawCall.fromAddress] = (senders[rawCall.fromAddress] || 0) + 1;
        for (const call of callDecoder.extractCalls(rawCall.callData, rawCall.callType)) {
            for (const order of call.orders) {
                orderCount++;
                info.orderCount++;
                const { feeRecipientAddress, makerAddress, senderAddress } = order;
                feeRecipients[feeRecipientAddress] = (feeRecipients[feeRecipientAddress] || 0) + 1;
                makers[makerAddress] = (makers[makerAddress] || 0) + 1;
            }
            info.fillCount += call.fills;
            info.updateCount += call.updates;
            methods[call.id] = (methods[call.id] || 0) + 1;
            process.stdout.write(
                `${++callCount} calls, ${orderCount} orders, ${Object.keys(callers).length} callers...\r`,
            );
        }
    });
    rl.on('close', async () => {
        process.stdout.write('\r\n');
        const lines = Object.values(callers).map(e => {
            if (ARGS.pretty) {
                return JSON.stringify(e, null, '  ');
            } else {
                return JSON.stringify(e);
            }
        }).join('\n');
        if (!ARGS.output) {
            console.log(lines);
        } else {
            await fs.writeFile(ARGS.output, lines, 'utf-8');
        }
    });
})();

async function loadMergedAbiFiles(files) {
    let abis = await Promise.all(files.map(f => fs.readFile(f, 'utf-8')));
    abis = abis.map(a => JSON.parse(a));
    abis = abis.map(a => _.isArray(a) ? a : a.compilerOutput.abi);
    return _.uniqBy(_.flatten(abis), a => a.name);
}

class CallDecoder {
    constructor(abi) {
        this._selectorsToMethodAbi = {};
        for (const method of abi) {
            if (method.type === 'function') {
                const signature = AbiEncoder.encodeFunctionSignature(method);
                this._selectorsToMethodAbi[signature] = method;
            }
        }
    }

    extractCalls(callData, callType) {
        const signature = callData.substr(0, 10);
        const rawArgs = '0x' + callData.substr(10);
        if (!(signature in this._selectorsToMethodAbi)) {
            throw new Error(`Unknown selector: ${signature}.`);
        }
        const method = this._selectorsToMethodAbi[signature];
        let args;
        try {
            args = CallDecoder.cleanArgs(
                AbiEncoder.decodeParameters(method.inputs, rawArgs),
            );
        } catch (err) {
            console.error(err);
            console.warn(method.name, rawArgs);
            args = {};
        }
        let orders = args.orders || [];
        let signatures = args.signatures || [];
        let fills = 0;
        let updates = 0;
        if (args.order) {
            orders = [ args.order ];
            signatures = [ args.signature ];
        } else if (args.leftOrder) {
            orders = [ args.leftOrder, args.rightOrder ];
            signatures = [ args.leftSignature, args.rightSignature ];
        }
        if (callType === 'call') {
            if (/fill/i.test(method.name) ||
                /buy/i.test(method.name) ||
                /sell/i.test(method.name) ||
                /match/i.test(method.name))
            {
                fills = orders.length;
            }
            updates = orders.length;
        }
        orders = orders.map(order => ({ ...order, hash: CallDecoder.hashOrder(order) }));
        const call = {
            id: method.name,
            orders,
            signatures,
            fills,
            updates,
        };
        if (method.name === 'executeTransaction') {
            return _.flatten([
                call,
                ...this.extractCalls(args.data, callType).map(c => ({...c, id: `tx_${c.id}`})),
            ]);
        }
        return [ call ];
    }

    static hashOrder(order) {
        // Any collision-proof, deterministic ID will do.
        const hasher = crypto.createHash('sha256');
        const keys = Object.keys(order).sort();
        const fields = _.zip(keys, keys.map(k =>order[k]));
        return '0x'+hasher.update(Buffer.from(JSON.stringify(fields))).digest('hex');
    }

    static cleanArgs(args) {
        if (_.isObject(args)) {
            if (_.every(Object.keys(args), k => !isNaN(parseInt(k)))) {
                // Array.
                return args.map(CallDecoder.cleanArgs);
            }
            return _.mapValues(
                _.omitBy(args, (v, k) => /^\d+$/.test(k) || k === '__length__'),
                v => CallDecoder.cleanArgs(v),
            );
        }
        return args;
    }
}
