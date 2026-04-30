import it.unimi.dsi.bits.TransformationStrategies;
import it.unimi.dsi.fastutil.io.BinIO;
import it.unimi.dsi.fastutil.longs.LongIterable;
import it.unimi.dsi.sux4j.mph.GV3CompressedFunction;
import it.unimi.dsi.sux4j.mph.GV4CompressedFunction;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Iterator;

/**
 * Builds a GOV compressed static function whose keys are the integers 0..n-1
 * stored as 4-byte arrays in native byte order. This lets the C benchmark
 * compute key[i] on the fly as (char*)&i with no separate key buffer.
 *
 * Usage: BuildCSF <3|4> <values_file> <output_file>
 *   values_file  – binary file of big-endian int64 values (one per key)
 *   output_file  – destination for the C-readable serialised function
 */
public class BuildCSF {
    public static void main(final String[] args) throws IOException {
        if (args.length != 3) {
            System.err.println("Usage: BuildCSF <3|4> <values_file> <output_file>");
            System.exit(1);
        }

        final int version = Integer.parseInt(args[0]);
        final String valuesFile = args[1];
        final String outputFile = args[2];

        // Derive n from the values file (8 bytes per long).
        final long n = new File(valuesFile).length() / 8;

        // Generate keys as 4-byte native-order integers: key[i] = (uint32_t)i.
        final Iterable<byte[]> keys = () -> new Iterator<byte[]>() {
            private long i = 0;
            @Override public boolean hasNext() { return i < n; }
            @Override public byte[] next() {
                return ByteBuffer.allocate(4).order(ByteOrder.nativeOrder()).putInt((int) i++).array();
            }
        };

        final LongIterable values = BinIO.asLongIterable(valuesFile);

        if (version == 3) {
            new GV3CompressedFunction.Builder<byte[]>()
                    .keys(keys)
                    .transform(TransformationStrategies.rawByteArray())
                    .values(values)
                    .build()
                    .dump(outputFile);
        } else {
            new GV4CompressedFunction.Builder<byte[]>()
                    .keys(keys)
                    .transform(TransformationStrategies.rawByteArray())
                    .values(values)
                    .build()
                    .dump(outputFile);
        }
    }
}
