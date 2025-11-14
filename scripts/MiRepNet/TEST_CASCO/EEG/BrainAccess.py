""" Interface with brainaccess core functions

This example uses asynchronous core functions

Change Bluetooth port according to your device
"""

import asyncio
from sys import platform

import brainaccess.core as bacore
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.core.eeg_manager import EEGManager

# Check version of the core
bacore.init(bacore.Version(2, 0, 0))
print(bacore.get_version())

# Como estamos en windows asignamos el puerto
port = "COM3" # NOTA: cambiar el puerto al q indique la interfaz del casco

# Start manager
with EEGManager() as mgr:

    # function to run when new data samples are ready
    def chunk_callback1(chunk, chunk_size):
        # WARNING: code running inside of callbacks may or may not be running in the reader thread.
        # This means that:
        # - While the callback is running, it might be blocking bluetooth communication
        # - It should be kept as short as possible
        # - It might need a lock/mutex if accessing a shared resource
        # If processing takes too long, getting the main thread's asyncio event loop and using call_soon_threadsafe is advisable.
        print(chunk)
        # for i in range(chunk_size):
        #     print(chunk[mgr.get_channel_index(eeg_channel.ELECTRODE_MEASUREMENT+0)][i])


    def chunk_callback(chunk, chunk_size):
        print("\n=== NEW CHUNK ===")
        print("chunk_size:", chunk_size)
        print("type(chunk):", type(chunk))
        print("len(chunk):", len(chunk))

        # Primer canal
        ch = chunk[0]
        print("type(chunk[0]):", type(ch))
        print("len(chunk[0]):", len(chunk[0]))
        print("chunk[0]:", ch)

        print("type(chunk[0][0]):", type(chunk[0][0]))

        # Si es iterable, imprime las primeras muestras
        try:
            print("First sample:", ch[0])
        except:
            pass

    # asynchronous data acquisition
    async def main():
        if await mgr.connect(port):

            # setting function to run when new data is ready
            mgr.set_callback_chunk(chunk_callback)

            # enabling channels to sample
            #mgr.set_channel_enabled(eeg_channel.SAMPLE_NUMBER, True)
            mgr.set_channel_enabled(eeg_channel.ELECTRODE_MEASUREMENT + 1, True)
            #mgr.set_channel_enabled(eeg_channel.DIGITAL_INPUT, True)

            # starting stream
            await mgr.start_stream()

            io_state = True
            for _ in range(100):
                # Send requests (these functions return futures, which must be awaited to get the actual result)
                fl = mgr.get_latency()
                fb = mgr.get_full_battery_info()

                # setting digital input on
                await mgr.set_io(0, io_state)
                io_state = not io_state
                #print(io_state)

                # Wait for responses
                l = await fl
                b = await fb

                # Print results
                print("Ping: " + str(l))
                print("Level: " + str(b.level))
                print("Current: " + str(b.current))
                print("Voltage: " + str(b.voltage))
                print("Health: " + str(b.health))

                # Wait for 1 second
                await asyncio.sleep(1)

            await mgr.stop_stream()
        mgr.disconnect()

    asyncio.run(main())

bacore.close()